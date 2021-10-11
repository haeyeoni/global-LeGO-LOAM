#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <cfloat>
 

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
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
#include <model/lidar_model.h>
#include <parameter.h>

using PointType = pcl::PointXYZ; // with intensity
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

namespace lego_loam
{

class ParticleFilter3D : public nodelet::Nodelet {

private:

    Parameters params_;
    ros::Subscriber subPointCloud;
    ros::Subscriber subInitialize;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubMapCloud;
    ros::Publisher pubMapCloudTemp;
    ros::Publisher pubParticles;
    ros::Publisher pubPose;
    ros::Publisher pubTransformedOdometry; // test;
    ros::Publisher pubTempCloud;
    
    
    pcl::PointCloud<PointType>::Ptr global_map_;
    pcl::PointCloud<PointType>::Ptr all_accum_pointcloud_;
    pcl::PointCloud<PointType>::Ptr local_accum_pointcloud_;
    std::vector<std_msgs::Header> accum_header_;

    bool loaded_map = false;  
    bool initialized = false;
    bool has_odom_;
    OdomModel::Ptr odom_model_;
    LidarModel::Ptr lidar_model_;

    ParticleFilter<PoseState>::Ptr pf_;    
    ChunkedKdtree<PointType>::Ptr map_kdtree_;
    
    // State represents
    PoseState odom_; // current odometry
    PoseState odom_prev_;
    PoseState state_prev_;

    // Time
    ros::Time odom_last_time_;
    ros::Time localized_last_;
    ros::Timer timer1;
    ros::Timer timer2;
    ros::Timer timer3;

    // TF 
    tf2_ros::Buffer tfbuf_;
    tf2_ros::TransformListener tfl_;
    tf2_ros::TransformBroadcaster tfb_;
    
    // transform
    Eigen::Affine3f tf_to_cam_ = Eigen::Affine3f::Identity();

    // flag
    size_t cnt_measure_;
    bool map_rotated_;

public:
    ParticleFilter3D(): tfl_(tfbuf_), cnt_measure_(0), map_rotated_(false) {}

    void onInit() 
    {       
        ROS_INFO("\033[1;32m---->\033[0m Particle Filter Started.");  

        ros::NodeHandle nh = getNodeHandle();
		ros::NodeHandle nhp = getPrivateNodeHandle();
        if (!params_.load(nhp))
        {
            ROS_ERROR("Failed to load parameters");
        }

        // load saved map
        global_map_.reset(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType> (params_.map_save_path_, *global_map_) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        ROS_INFO_ONCE("Map loaded");
        loaded_map = true;
        
        // reset particle filter and model 
        pf_.reset(new ParticleFilter<PoseState>(params_.num_particles_, params_.sampling_covariance_)); 
        odom_model_.reset(new OdomModel(params_.odom_lin_err_sigma_, params_.odom_ang_err_sigma_, params_.odom_lin_err_tc_, params_.odom_ang_err_tc_,
                                        params_.max_z_pose_, params_.min_z_pose_));
        lidar_model_.reset(new LidarModel(params_.num_points_, params_.num_points_global_, params_.max_search_radius_, params_.min_search_radius_, 
                                          params_.clip_near_sq_, params_.clip_far_sq_, params_.clip_z_min_, params_.clip_z_max_,
                                          params_.perform_weighting_ratio_, params_.max_weight_ratio_, params_.max_weight_, params_.normal_search_range_, params_.odom_lin_err_sigma_));
        
        map_kdtree_.reset(new ChunkedKdtree<PointType>(params_.map_chunk_, params_.max_search_radius_));
        map_kdtree_->setEpsilon(params_.map_grid_min_ / 16);
        
        // ros subscriber & publisher

        if (params_.use_initial_pose_)
            timer3 = nhp.createTimer(ros::Duration(0.1), boost::bind(&ParticleFilter3D::handleInitializeData1, this, _1));
        else
            subInitialize = nh.subscribe("/initialize_data", 10, &ParticleFilter3D::handleInitializeData, this);

        subPointCloud = nh.subscribe("/velodyne_points", 10, &ParticleFilter3D::handlePointCloud, this);
        subLaserOdometry = nh.subscribe("/integrated_to_init", 100, &ParticleFilter3D::handleLaserOdometry, this);

        pubPose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 5, true);
        pubParticles = nh.advertise<geometry_msgs::PoseArray>("/particles", 100, true);
        pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
        pubTempCloud = nh.advertise<sensor_msgs::PointCloud2>("/transformed_cloud", 10);
        pubTransformedOdometry = nh.advertise<nav_msgs::Odometry> ("/tranformed_odom", 5);        
    
        timer1 = nhp.createTimer(ros::Duration(1.0), boost::bind(&ParticleFilter3D::handleMapCloud, this, _1));
        timer2 = nhp.createTimer(ros::Duration(0.01), boost::bind(&ParticleFilter3D::publishParticles, this, _1));    
    }

    void handleMapCloud(const ros::TimerEvent& event)
    {
        // Downsampling the pointcloud map
        if (!loaded_map)
            return;
        pcl::VoxelGrid<PointType> ds;
        ds.setInputCloud(global_map_);
        ds.setLeafSize(params_.map_downsample_x_, params_.map_downsample_y_, params_.map_downsample_z_);
        ds.filter(*global_map_);

        // transform map to map frames (camera_init -> map)
            
        if (!map_rotated_)
        {        
            tf_to_cam_.translation() << 0.0, 0.0, 0.0;
            
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(params_.map_yaw_, Eigen::Vector3f::UnitZ()));
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(params_.map_pitch_, Eigen::Vector3f::UnitY()));
            tf_to_cam_.rotate(Eigen::AngleAxis<float>(params_.map_roll_, Eigen::Vector3f::UnitX()));

            pcl::transformPointCloud (*global_map_, *global_map_, tf_to_cam_);

            map_kdtree_->setInputCloud(global_map_);
            map_rotated_ = true;
        }

        lidar_model_->setKdtree(map_kdtree_);
        // publish map
        sensor_msgs::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*global_map_, map_cloud_msg);
        map_cloud_msg.header.stamp = ros::Time::now();
        map_cloud_msg.header.frame_id = "map";
        pubMapCloud.publish(map_cloud_msg);  
    }

    void handleLaserOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg)
    {
        ROS_INFO_ONCE("odom received");

        tf::Quaternion q_rot = tf::createQuaternionFromRPY(params_.rot_x_, params_.rot_y_, params_.rot_z_);

        tf::Quaternion q_orientation, q_orientation_rot;	
        tf::quaternionMsgToTF(odom_msg->pose.pose.orientation, q_orientation);
        q_orientation_rot = q_orientation * q_rot ;
        q_orientation_rot.normalize();

        odom_ =  PoseState(Vec3(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z),
                        Quat(q_orientation_rot.x(), q_orientation_rot.y(), q_orientation_rot.z(), q_orientation_rot.w()) );

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

    void handlePointCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
    {
        if(!initialized)
            return;
        
        ROS_INFO_ONCE("cloud received");

        if(!loaded_map)
            return;
        all_accum_pointcloud_.reset(new pcl::PointCloud<PointType>);
        local_accum_pointcloud_.reset(new pcl::PointCloud<PointType>);
        local_accum_pointcloud_->header.frame_id = "laser_odom";
        accum_header_.clear();
        if (accumCloud(cloud_msg)) // accumulate the transformed point cloud
            measure();
    }

    bool accumCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
    {
        ROS_INFO_ONCE("accumCloud called");
        sensor_msgs::PointCloud2 transformed_pointcloud;
        try
        {
            const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform("laser_odom",
                                                                                    cloud_msg->header.frame_id,
                                                                                    ros::Time::now(),
                                                                                    ros::Duration(1)); // odom to cloud tf
            tf2::doTransform(*cloud_msg, transformed_pointcloud, trans);   
        }
        catch(const std::exception& e)
        {
            ROS_INFO("Failed to transform pointcloud in accumCloud function: %s", e.what());
            return false;
        }
        pcl::PointCloud<PointType>::Ptr temp_pointcloud(new pcl::PointCloud<PointType>);
        pcl::fromROSMsg(transformed_pointcloud, *temp_pointcloud);

        *local_accum_pointcloud_ += *temp_pointcloud;
        accum_header_.push_back(cloud_msg->header);
        return true;      
    }

    void measure()
    {
        ROS_INFO_ONCE("measure called");
        cnt_measure_++;
        if (cnt_measure_ % static_cast<size_t>(params_.skip_measure_) != 0)
        {
            return;
        }

        const std_msgs::Header& header = accum_header_.back(); // last header (most recent)
        // 1. Accumulate current tranformed Pointcloud in local_accum_pointcloud_
        try 
        {
            const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
                "base_link", local_accum_pointcloud_->header.frame_id, header.stamp, ros::Duration(1)); 
                    // tf between base to point cloud at header time (current)
            const Eigen::Affine3f trans_eigen =  Eigen::Translation3f(trans.transform.translation.x, 
                                                                    trans.transform.translation.y, 
                                                                    trans.transform.translation.z) *
                                                Eigen::Quaternionf(trans.transform.rotation.w, 
                                                                    trans.transform.rotation.x, 
                                                                    trans.transform.rotation.y, 
                                                                    trans.transform.rotation.z);
            pcl::transformPointCloud(*local_accum_pointcloud_, *local_accum_pointcloud_, trans_eigen); //local accum: current pointcloud -> transform to the base
            
            sensor_msgs::PointCloud2 pointcloud_msg;    
            pcl::toROSMsg(*local_accum_pointcloud_, pointcloud_msg);
            pointcloud_msg.header = header;
            pubTempCloud.publish(pointcloud_msg);
        }
        catch (tf2::TransformException& e)
        {
            ROS_INFO("Failed to transform pointcloud in measure function: %s", e.what());
            return;
        }

        // 2. Calculate the origins history of pointcloud 
        std::vector<Vec3> origins; // list of the origin of the previous lidar
        for (auto& h : accum_header_) 
        {
            try 
            {
                const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
                    "base_link", header.stamp, h.frame_id, h.stamp, "laser_odom");
                    // tf between base to pc at header time with odom fixed frame (history)
                    // lookupTransform(target_frame, target_time, source_frame, source_time, fixed_frame, timeout=0.0)
                origins.push_back(Vec3(trans.transform.translation.x,
                                        trans.transform.translation.y,
                                        trans.transform.translation.z));
            }
            catch (tf2::TransformException& e)
            {
                ROS_INFO("Failed to transform pointcloud: %s", e.what());
                return;
            }
        }
    
        // 3. Down sampling the current accumulated point cloud (pc_local_accum)
        const auto ts = boost::chrono::high_resolution_clock::now();
        pcl::PointCloud<PointType>::Ptr pc_local_full(new pcl::PointCloud<PointType>());
        pcl::VoxelGrid<PointType> ds;
        ds.setInputCloud(local_accum_pointcloud_);
        ds.setLeafSize(params_.downsample_x_, params_.downsample_y_, params_.downsample_z_);
        ds.filter(*pc_local_full);
        
        pcl::PointCloud<PointType>::Ptr pc_locals; 
        lidar_model_->setParticleNumber(params_.num_particles_, pf_->getParticleSize()); // Modify number of particles
        
        // 4. Calculate the particle filter statistics: mean and covariance 
        const PoseState prev_mean = pf_->mean();
        const float cov_ratio = std::max(0.1f, static_cast<float>(params_.num_particles_) / pf_->getParticleSize());
        const std::vector<PoseState> prev_cov = pf_->covariance(cov_ratio); 
        
        // 5. Random sample with normal distribution
        auto sampler = std::dynamic_pointer_cast<PointCloudSampler<PointType>>(lidar_model_->getSampler());    
        sampler->setParticleStatistics(prev_mean, prev_cov);  
        pc_locals = lidar_model_->filter(pc_local_full); 
    
        if (pc_locals->size() == 0)
        {
            ROS_ERROR("All points are filtered out.");
            return;
        }
        
        // 6. Correction Step with Lidar model
        float match_ratio_max = lidar_model_->measureCorrect(pf_, pc_locals, origins);
        // std::cout<<"match ratio: "<< match_ratio_max <<std::endl;
        if (match_ratio_max < params_.match_ratio_thresh_)
        {
            ROS_WARN_THROTTLE(3.0, "Low match ratio. Expansion resetting"); //every 3.0 seconds
            pf_->generateNoise(PoseState(Vec3(params_.expansion_var_x_, params_.expansion_var_y_, params_.expansion_var_z_),
                                Vec3(params_.expansion_var_roll_, params_.expansion_var_pitch_, params_.expansion_var_yaw_)));
        }
        pf_->resample(PoseState(
            Vec3(params_.resample_var_x_,
                params_.resample_var_y_,
                params_.resample_var_z_),
            Vec3(params_.resample_var_roll_,
                params_.resample_var_pitch_,
                params_.resample_var_yaw_)));
        pf_-> updateNoise(params_.odom_err_lin_lin_, params_.odom_err_lin_ang_,
                        params_.odom_err_ang_lin_, params_.odom_err_ang_ang_);

        auto biased_mean = pf_->biasedMean(odom_prev_, params_.num_particles_, params_.bias_var_dist_, params_.bias_var_ang_); 
        biased_mean.rot_.normalize();

        assert(std::isfinite(biased_mean.pose_.x_) && std::isfinite(biased_mean.pose_.y_) && std::isfinite(biased_mean.pose_.z_) &&
            std::isfinite(biased_mean.rot_.x_) && std::isfinite(biased_mean.rot_.y_) && std::isfinite(biased_mean.rot_.z_) && std::isfinite(biased_mean.rot_.w_));
        
        publishPose(biased_mean, header);  


        // 7. Publish map tf
        ros::Time localized_current = ros::Time::now();
        float dt = (localized_current - localized_last_).toSec();
        if (dt > 1.0) dt = 1.0;
        else if ( dt < 0.0) dt = 0.0;
        localized_last_ = localized_current;   
        Vec3 map_pose = biased_mean.pose_ - biased_mean.rot_.inv() * odom_.pose_;
        Quat map_rot = biased_mean.rot_ * odom_.rot_.inv();
        
    }

    void handleInitializeData(const std_msgs::Float32MultiArray::Ptr& initial_msg)
    {  
        int length = (int) initial_msg->layout.dim[1].size;
            
        if(!initialized && length > 0)
        {
            std::cout<<"**********Initialization************"<<std::endl;
            float similaritySum = initial_msg->layout.dim[0].size;
            pf_->init(length, similaritySum, initial_msg->data, params_.initial_yaw_); 
            initialized = true;
        }
    }

    void handleInitializeData1(const ros::TimerEvent& event)
    {  
        std::vector<float> pose; // x pose, y pose, similarity term
        pose.push_back(params_.initial_x_);
        pose.push_back(params_.initial_y_);
        pose.push_back(1.0);        
        
        if(!initialized)
        {
            std::cout<<"**********Initialization************"<<std::endl;
            pf_->init(1, 1.0, pose, params_.initial_yaw_);
            initialized = true;
        }
    }

    void publishParticles(const ros::TimerEvent& event)
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

    void publishPose(PoseState& biased_mean, const std_msgs::Header& header)
    {
        geometry_msgs::PoseWithCovarianceStamped pose;

        geometry_msgs::TransformStamped trans;
        trans.header.frame_id = params_.frame_ids_["map"];
        trans.child_frame_id = params_.frame_ids_["floor"];
        trans.transform.translation = tf2::toMsg(tf2::Vector3(0.0, 0.0, biased_mean.pose_.z_));
        trans.transform.rotation = tf2::toMsg(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));
        pose.header.stamp = header.stamp;
        pose.header.frame_id = trans.header.frame_id;
        pose.pose.pose.position.x = biased_mean.pose_.x_;
        pose.pose.pose.position.y = biased_mean.pose_.y_;
        pose.pose.pose.position.z = biased_mean.pose_.z_;
        pose.pose.pose.orientation.x = biased_mean.rot_.x_;
        pose.pose.pose.orientation.y = biased_mean.rot_.y_;
        pose.pose.pose.orientation.z = biased_mean.rot_.z_;
        pose.pose.pose.orientation.w = biased_mean.rot_.w_;
        auto cov = pf_->covariance(std::max(0.1f, static_cast<float>(params_.num_particles_)/pf_->getParticleSize()));
        for (size_t i = 0; i < 36; i ++)
            pose.pose.covariance[i] = cov[i / 6][i % 6];
        pubPose.publish(pose);
    }
};
}

PLUGINLIB_EXPORT_CLASS(lego_loam::ParticleFilter3D, nodelet::Nodelet)
