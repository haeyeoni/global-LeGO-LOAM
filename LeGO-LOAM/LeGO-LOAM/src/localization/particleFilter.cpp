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

#include <pf.h>
#include <state/state_rep.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

using PointType = pcl::PointXYZI; // with intensity
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

class particleFilter3D{

private:
    ros::NodeHandle nh;
    ros::Subscriber subPointCloud;
    ros::Subscriber subInitialize;
    ros::Publisher pubMapCloud;
    ros::Publisher pubParticles;
    ros::Publisher pubPose;
    
    
    pcl::PointCloud<PointType>::Ptr globalMap;

    bool loaded_map = false;  
    bool initialized = false;
    int num_particles;
    double sampling_covariance;
    ParticleFilter<PoseState>::Ptr pf_;


public:
    particleFilter3D():nh("~")
    {   // parameter
        nh.param<int>("num_particles", num_particles, 100); 
        nh.param<double>("sampling_covariance", sampling_covariance, 0.1); 
        
        // load saved map
        globalMap.reset(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType> ("/home/haeyeon/Cocel/lego_loam_map.pcd", *globalMap) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        loaded_map = true;
        std::cout<< "Loaded Map Cloud"<<std::endl; 
        
        // reset pf

        pf_.reset(new ParticleFilter<PoseState>(num_particles, sampling_covariance)); 
        
        subInitialize = nh.subscribe("/initialize_data", 10, &particleFilter3D::handleInitializeData, this);
        subPointCloud = nh.subscribe("velodyne_points", 10, &particleFilter3D::handlePointCloud, this);
        
        pubPose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 5, true);
        pubParticles = nh.advertise<geometry_msgs::PoseArray>("/particles", 1, true);
        pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
        
    }
    void handlePointCloud(const sensor_msgs::PointCloud2::Ptr& pcl_msg)
    {

    }

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

    void publishMap()
    {
        if (loaded_map)
        {
            sensor_msgs::PointCloud2 mapCloudMsg;
            pcl::toROSMsg(*globalMap, mapCloudMsg);
            mapCloudMsg.header.stamp = ros::Time::now();
            mapCloudMsg.header.frame_id = "/camera_init";
            pubMapCloud.publish(mapCloudMsg);
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
        PF.publishMap();
        PF.publishParticles();
    }

    return 0;
}

