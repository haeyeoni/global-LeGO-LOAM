#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/chrono.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <pf.h>
#include <state/state_rep.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <chunked_kdtree.h>
#include <model/odom_model.h>
// #include <model/lidar_model.h>


using PointType = pcl::PointXYZI; // with intensity
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

class particleFilter3D{

private:
    ros::NodeHandle nh;
    ros::Subscriber subPointCloud;
    ros::Subscriber subInitialize;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubMapCloud;
    ros::Publisher pubParticles;
    ros::Publisher pubPose;
    ros::Publisher pubTransformedOdometry; // test;
    
    
    pcl::PointCloud<PointType>::Ptr global_map_;

    bool loaded_map = false;  
    bool initialized = false;
    bool has_odom_;
    OdomModel::Ptr odom_model_;
    // LidarModel::Ptr lidar_model_;

    ParticleFilter<PoseState>::Ptr pf_;    
    ChunkedKdtree<PointType>::Ptr map_kdtree_;
    
    // State represents
    PoseState odom_; // current odometry
    PoseState odom_prev_;
    PoseState state_prev_;

    // Time
    ros::Time odom_last_time_;

    // transform
    Eigen::Affine3f tf_to_cam_ = Eigen::Affine3f::Identity();

    // parameter
    int num_particles_;
    double sampling_covariance_;
    double map_grid_min_, map_grid_max_;
    double map_chunk_;
    double max_search_radius_;
    double map_downsample_x_, map_downsample_y_, map_downsample_z_;

public:
    particleFilter3D():nh("~")
    {   // parameter
        nh.param("num_particles", num_particles_, 100); 
        nh.param("sampling_covariance", sampling_covariance_, 0.1); 
        nh.param("map_chunk", map_chunk_, 20.0); 
        nh.param("max_search_radius", max_search_radius_, 0.2); 
                
        nh.param("map_downsample_x", map_downsample_x_, 0.1);
        nh.param("map_downsample_y", map_downsample_y_, 0.1);
        nh.param("map_downsample_z", map_downsample_z_, 0.1);
        map_grid_min_ = std::min(std::min(map_downsample_x_, map_downsample_y_), map_downsample_z_);
        map_grid_max_ = std::max(std::max(map_downsample_x_, map_downsample_y_), map_downsample_z_);

        // load saved map
        global_map_.reset(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType> ("/home/haeyeon/Cocel/lego_loam_map.pcd", *global_map_) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        loaded_map = true;
        
        // reset particle filter and model 
        pf_.reset(new ParticleFilter<PoseState>(num_particles_, sampling_covariance_)); 
        odom_model_.reset(new OdomModel(0.0, 0.0, 0.0, 0.0));
        // lidar_model_.reset(new LidarModel());
        map_kdtree_.reset(new ChunkedKdtree<PointType>(map_chunk_, max_search_radius_, map_grid_min_ /16));        


        // ros subscriber & publisher
        subInitialize = nh.subscribe("/initialize_data", 10, &particleFilter3D::handleInitializeData, this);
        subPointCloud = nh.subscribe("/velodyne_points", 10, &particleFilter3D::handlePointCloud, this);
        subLaserOdometry = nh.subscribe("/laser_odom_to_init", 10, &particleFilter3D::handleLaserOdometry, this);

        pubPose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 5, true);
        pubParticles = nh.advertise<geometry_msgs::PoseArray>("/particles", 1, true);
        pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
        pubTransformedOdometry = nh.advertise<nav_msgs::Odometry> ("/tranformed_odom", 5);
    }

    void handleMapCloud()
    {
        ROS_INFO_ONCE("Map loaded");

        // Downsampling the pointcloud map
        pcl::VoxelGrid<PointType> ds;
        ds.setInputCloud(global_map_);
        ds.setLeafSize(map_downsample_x_, map_downsample_y_, map_downsample_z_);
        ds.filter(*global_map_);

        // transform map to map frames (camera_init -> map)
        pcl::PointCloud<PointType>::Ptr map_transformed(new pcl::PointCloud<PointType>());
        tf::TransformListener tf_listener;
        tf::StampedTransform transform;
        if (!tf_listener.waitForTransform("/map", "/camera_init", ros::Time(0), ros::Duration(0.5), ros::Duration(0.01))) 
        {
            ROS_ERROR("Unable to get pose from TF");
            return;
        }
        try 
        {
            tf_listener.lookupTransform("/map", "/camera_init", ros::Time(0), transform);
            // angle
			tf2::Quaternion orientation (transform.getRotation().x(), transform.getRotation().y(), transform.getRotation().z(), transform.getRotation().w());
			tf2::Matrix3x3 m(orientation);
			double roll, pitch, yaw;
			m.getRPY(roll, pitch, yaw);
            tf_to_cam_.translation() << transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z();
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(roll, Eigen::Vector3f::UnitX()));
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(pitch, Eigen::Vector3f::UnitY()));
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(yaw, Eigen::Vector3f::UnitZ()));
            pcl::transformPointCloud (*global_map_, *map_transformed, tf_to_cam_);
            map_kdtree_->setInputCloud(map_transformed);
        }
        catch (const tf::TransformException &e) {
            ROS_ERROR("%s",e.what());
        } 			
        // lidar_model_->setKdtree(map_kdtree_);

        // publish map
        sensor_msgs::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*global_map_, map_cloud_msg);
        map_cloud_msg.header.stamp = ros::Time::now();
        map_cloud_msg.header.frame_id = "/camera_init";
        pubMapCloud.publish(map_cloud_msg);
    }

    void handleLaserOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg)
    {
        ROS_INFO_ONCE("odom received");

        Quat trans1(Vec3(0,1,0), -M_PI/2);
        Quat trans2(Vec3(1,0,0), -M_PI/2);
        
        Quat odom_ori(odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y,
                     odom_msg->pose.pose.orientation.z, odom_msg->pose.pose.orientation.w);
        
        Quat trans_ori = odom_ori * trans1 * trans2;

        odom_ =  PoseState(Vec3(-odom_msg->pose.pose.position.x, 
                            odom_msg->pose.pose.position.y,
                            odom_msg->pose.pose.position.z),
                        (Quat(trans_ori[0], trans_ori[1], trans_ori[2], trans_ori[3])));

        if (!has_odom_) // if this is initial odom
        {
            odom_prev_ = odom_;
            odom_last_time_ = odom_msg->header.stamp;
            has_odom_ = true;
            return;
        }
        const float dt = (odom_msg->header.stamp - odom_last_time_).toSec();    
        if (dt < 0.0 || dt > 5.0)
        {
            ROS_WARN("Detected time jump in odometry. %f", dt);
            has_odom_ = false;
            return;
        }
        else if (dt > 0.05)
        {
            odom_model_->setOdoms(odom_prev_, odom_, dt);
            odom_model_->motionPredict(pf_);
            odom_last_time_ = odom_msg->header.stamp;
            odom_prev_ = odom_;
        }  
    }


    void handlePointCloud(const sensor_msgs::PointCloud2::ConstPtr& odom_msg)
    {}

    void handleInitializeData(const std_msgs::Float32MultiArray::Ptr& initial_msg)
    {
        int length = (int) initial_msg->layout.dim[1].size;
            
        if(!initialized && length > 0)
        {
            std::cout<<"**********Initialization************"<<std::endl;
            float similaritySum = initial_msg->layout.dim[0].size;
            pf_->init(length, similaritySum, initial_msg->data);   
            initialized = true;
        }
    }


    void publishParticles()
    {
        if(initialized)
        {
            geometry_msgs::PoseArray particles;
            particles.header.stamp = ros::Time::now();
            particles.header.frame_id = "map";
            
            for (size_t i = 0; i < pf_->getParticleSize(); i++)
            {
                geometry_msgs::Pose particle;
                auto p = pf_->getParticle(i);
                p.rot_.normalize();
                particle.position.x = p.pose_.x_;
                particle.position.y = p.pose_.y_;
                particle.position.z = p.pose_.z_;
                particle.orientation.x = p.rot_.x_;
                particle.orientation.y = p.rot_.y_;
                particle.orientation.z = p.rot_.z_;
                particle.orientation.w = p.rot_.w_;
                particles.poses.push_back(particle);       
            }    
            
            pubParticles.publish(particles);        
        }
    }

};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("Global Localization Started.");

    particleFilter3D PF;
    
    while (ros::ok())
    {
        ros::spinOnce();
        PF.handleMapCloud();
        PF.publishParticles();
    }

    return 0;
}

