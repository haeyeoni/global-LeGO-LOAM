#include <amcl3d/amcl3d_node.h>

namespace amcl3d
{
using PointType = pcl::PointXYZ; // with intensity
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

void AMCL3DNode::handleInitialPose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &pose_msg)
{
    const double norm = pose_msg->pose.pose.orientation.x * pose_msg->pose.pose.orientation.x +
                        pose_msg->pose.pose.orientation.y * pose_msg->pose.pose.orientation.y +
                        pose_msg->pose.pose.orientation.z * pose_msg->pose.pose.orientation.z +
                        pose_msg->pose.pose.orientation.w * pose_msg->pose.pose.orientation.w;
    
    if (std::abs(norm - 1.0) > 0.1)
    {
        ROS_ERROR("The orientation should be unit quaternion.");
    }
    geometry_msgs::PoseStamped pose_in, pose;
    pose_in.header = pose_msg->header;
    pose_in.pose = pose_msg->pose.pose;

    try 
    {
        const geometry_msgs::TransformStamped trans 
                        = tfbuf_.lookupTransform(params_.frame_ids_["map"], pose_in.header.frame_id, pose_in.header.stamp, ros::Duration(1.0));
                        // tf between map and pose
        tf2::doTransform(pose_in, pose, trans);
    }
    catch (tf2::TransformException& e) 
    {
        return;
    }
    const PoseState mean(Vec3(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z), 
                         Quat(pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w));
    
    // multivariate covariance
    const size_t dim = 6; //x,y,z,roll,pitch,yaw
    Matrix cov_matrix(dim, dim);
    
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            cov_matrix(i, j) = pose_msg->pose.covariance[i + j*dim];
            
    pf_->init_with_initialpose(mean, cov_matrix);
    pc_update_.reset();
    publishParticles();
}    
    
void AMCL3DNode::handleMapCloud(const sensor_msgs::PointCloud2::ConstPtr &map_msg)
{
    ROS_INFO_ONCE("map received");
    pcl::PointCloud<PointType>::Ptr pc_tmp(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*map_msg, *pc_tmp);
    const ros::Time map_stamp = (map_msg->header.stamp != ros::Time()) ? map_msg->header.stamp : ros::Time::now();
    pcl_conversions::toPCL(map_stamp, pc_tmp->header.stamp);
    pc_map_.reset(new pcl::PointCloud<PointType>);
    pc_map2_.reset();
    pc_update_.reset();

    // Downsampling the pointcloud map
    pcl::VoxelGrid<PointType> ds;
    ds.setInputCloud(pc_tmp);
    ds.setLeafSize(params_.map_downsample_x_, params_.map_downsample_y_, params_.map_downsample_z_);
    ds.filter(*pc_map_);

    // Reset the accumulated Clouds 
    pc_all_accum_.reset(new pcl::PointCloud<PointType>());
    pc_local_accum_.reset(new pcl::PointCloud<PointType>());
    pc_local_accum_->header.frame_id = params_.frame_ids_["odom"];
    pc_accum_header_.clear();
    
    has_map_ = true;
    mapUpdater(ros::TimerEvent()); // put map pointcloud into kdtree
}

void AMCL3DNode::mapUpdater(const ros::TimerEvent& event)
{
    if (has_map_)
    {
        const auto ts = boost::chrono::high_resolution_clock::now();
        if (pc_update_)
        {
            if(!pc_map2_)
                pc_map2_.reset(new pcl::PointCloud<PointType>());
            *pc_map2_ = *pc_map_ + *pc_update_;
            ROS_DEBUG("map updated");
            pc_update_.reset();
            pcl_conversions::toPCL(ros::Time::now(), pc_map2_->header.stamp);
        }
        else
        {
            if(pc_map2_)
                return;
            pc_map2_ = pc_map_;
        }
        kdtree_->setInputCloud(pc_map2_); // insert updated map into chunked kdtree
        lidar_model_->setKdtree(kdtree_);
        sensor_msgs::PointCloud2 out;
        pcl::toROSMsg(*pc_map2_, out);
        pub_mapcloud_.publish(out); //publish updated map
    }
}   

void AMCL3DNode::handleOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    ROS_INFO_ONCE("odom received");
    
    odom_ =  PoseState(Vec3(odom_msg->pose.pose.position.x,
                            odom_msg->pose.pose.position.y,
                            odom_msg->pose.pose.position.z),
                        (Quat(odom_msg->pose.pose.orientation.x,
                            odom_msg->pose.pose.orientation.y,
                            odom_msg->pose.pose.orientation.z,
                            odom_msg->pose.pose.orientation.w))); // save odom message to class variable
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

void AMCL3DNode::handleImu(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
    const Vec3 acc = Vec3(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    
    if (!has_imu_)
    {
        // acc_filter_->set(Vec3()); // initialize lpf filter
        // gyr_filter_->set(Vec3());
        imu_last_ = imu_msg->header.stamp;
        has_imu_ = true;
        return;
    }
    float dt = (imu_msg->header.stamp - imu_last_).toSec();
    
    if (dt < 0.0 || dt > 5.0)
    {
        has_imu_ = false;
        return;
    }
    else if (dt > 0.05)
    {
        Vec3 acc_measure = acc.normalized();
        // Vec3 gyr_measure = gyr.normalized();
        try
        {
            geometry_msgs::Vector3 acc_in, acc_out;
            acc_in.x = acc_measure.x_;
            acc_in.y = acc_measure.y_;
            acc_in.z = acc_measure.z_;
            // gyr_in.x = gyr_measure.x_;
            // gyr_in.y = gyr_measure.y_;
            // gyr_in.z = gyr_measure.z_;

            const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
                params_.frame_ids_["base_link"], imu_msg->header.frame_id, ros::Time(0));
            
            tf2::doTransform(acc_in, acc_out, trans);
            // tf2::doTransform(gyr_in, gyr_out, trans);

            acc_measure = Vec3(acc_out.x, acc_out.y, acc_out.z); // transformed imu linear acceleration
            imu_quat_.x_ = imu_msg->orientation.x;
            imu_quat_.y_ = imu_msg->orientation.y;
            imu_quat_.z_ = imu_msg->orientation.z;
            imu_quat_.w_ = imu_msg->orientation.w;
            Vec3 axis;
            float angle;
            imu_quat_.getAxisAng(axis, angle);

            axis = Quat(trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w) * axis;
            
            imu_quat_.setAxisAng(axis, angle);
            
            // gyr_measure = Vec3(gyr_out.x, gyr_out.y, gyr_out.z); // transformed imu angular velocity
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return;
        }

        imu_model_->setAccMeasure(acc_measure); // set Current imu msg 
        // imu_model_->setGyrMeasure(gyr_measure);
        
        imu_model_->imuPredict(pf_);
        imu_last_ = imu_msg->header.stamp;
        
        // nav_msgs::Odometry::Ptr odom(new nav_msgs::Odometry);
        // odom->header.frame_id = params_.frame_ids_["base_link"];
        // odom->header.stamp = imu_msg->header.stamp;
        // odom->pose.pose.orientation.x = imu_quat_.x_;
        // odom->pose.pose.orientation.y = imu_quat_.y_;
        // odom->pose.pose.orientation.z = imu_quat_.z_;
        // odom->pose.pose.orientation.w = imu_quat_.w_;
        // handleOdometry(odom);
    }
}

void AMCL3DNode::handlePointCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    ROS_INFO_ONCE("points received");
    
    if (!has_map_)
        return;
    pc_all_accum_.reset(new pcl::PointCloud<PointType>);
    pc_local_accum_.reset(new pcl::PointCloud<PointType>);
    pc_local_accum_->header.frame_id = params_.frame_ids_["odom"];
    pc_accum_header_.clear();
    if (accumCloud(cloud_msg)) // accumulate the transformed point cloud
        measure(); 
}

bool AMCL3DNode::accumCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    sensor_msgs::PointCloud2 pc_transformed;    
    try
    {
        const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform( params_.frame_ids_["odom"],
                                                                                cloud_msg->header.frame_id,
                                                                                cloud_msg->header.stamp,
                                                                                ros::Duration(1)); // odom to cloud tf
        tf2::doTransform(*cloud_msg, pc_transformed, trans);                                                                                 
    }
    catch (tf2::TransformException& e)
    {
        ROS_INFO("Failed to transform pointcloud in accumCloud function: %s", e.what());
        return false;
    }
    pcl::PointCloud<PointType>::Ptr pc_tmp(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(pc_transformed, *pc_tmp);

    // TODO: ADD LABEL 
    *pc_local_accum_ += *pc_tmp;
    pc_accum_header_.push_back(cloud_msg->header);
    return true;
}

void AMCL3DNode::measure()
{
    const std_msgs::Header& header = pc_accum_header_.back(); // last header (most recent)
    // 1. Accumulate current tranformed Pointcloud in pc_local_accum
    try 
    {
        const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
            params_.frame_ids_["base_link"], pc_local_accum_->header.frame_id, header.stamp, ros::Duration(1)); 
                // tf between base to pc at header time (current)
        const Eigen::Affine3f trans_eigen =  Eigen::Translation3f(trans.transform.translation.x, 
                                                               trans.transform.translation.y, 
                                                                trans.transform.translation.z) *
                                            Eigen::Quaternionf(trans.transform.rotation.w, 
                                                                trans.transform.rotation.x, 
                                                                trans.transform.rotation.y, 
                                                                trans.transform.rotation.z);
        pcl::transformPointCloud(*pc_local_accum_, *pc_local_accum_, trans_eigen); //local accum: current pointcloud -> transform to the base
        
        sensor_msgs::PointCloud2 pc_publish;    
        pcl::toROSMsg(*pc_local_accum_, pc_publish);
        pc_publish.header = header;
        pub_tempcloud_.publish(pc_publish);

    }
    catch (tf2::TransformException& e)
    {
        ROS_INFO("Failed to transform pointcloud in measure function: %s", e.what());
        return;
    }
    
    // 2. Calculate the origins history of pointcloud 
    std::vector<Vec3> origins; // list of the origin of the previous lidar
    for (auto& h : pc_accum_header_) 
    {
        try 
        {
            const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
                params_.frame_ids_["base_link"], header.stamp, h.frame_id, h.stamp, params_.frame_ids_["odom"]);
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

    ROS_INFO("Flag1");
    // 3. Down sampling the current accumulated point cloud (pc_local_accum)
    const auto ts = boost::chrono::high_resolution_clock::now();
    pcl::PointCloud<PointType>::Ptr pc_local_full(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> ds;
    ds.setInputCloud(pc_local_accum_);
    ds.setLeafSize(params_.downsample_x_, params_.downsample_y_, params_.downsample_z_);
    ds.filter(*pc_local_full);
    
    pcl::PointCloud<PointType>::Ptr pc_locals; 
    lidar_model_->setParticleNumber(params_.num_particles_, pf_->getParticleSize()); // Modify number of particles
    
    ROS_INFO("Flag2");
    // 4. Calculate the particle filter statistics: mean and covariance 
    const PoseState prev_mean = pf_->mean();
    const float cov_ratio = std::max(0.1f, static_cast<float>(params_.num_particles_) / pf_->getParticleSize());
    const std::vector<PoseState> prev_cov = pf_->covariance(cov_ratio); 
    
    // 5. Random sample with normal distribution
    auto sampler = std::dynamic_pointer_cast<PointCloudSampler<PointType>>(lidar_model_->getSampler());    
    sampler->setParticleStatistics(prev_mean, prev_cov);  
    pc_locals = lidar_model_->filter(pc_local_full); 
    
    ROS_INFO("Flag3");
    if (pc_locals->size() == 0)
    {
        ROS_ERROR("All points are filtered out.");
        return;
    }
    
    // 6. Correction Step with Lidar model
    float match_ratio_max = lidar_model_->measureCorrect(pf_, pc_locals, origins);
    
    ROS_INFO("Flag3.5");
    
    if (match_ratio_max < params_.match_ratio_thresh_)
    {
        ROS_WARN_THROTTLE(3.0, "Low match ratio. Expansion resetting"); //every 3.0 seconds
        pf_->generateNoise(PoseState(Vec3(params_.expansion_var_x_, params_.expansion_var_y_, params_.expansion_var_z_),
                             Vec3(params_.expansion_var_roll_, params_.expansion_var_pitch_, params_.expansion_var_yaw_)));
    }

    ROS_INFO("Flag4");
    pf_->resample(PoseState(
        Vec3(params_.resample_var_x_,
             params_.resample_var_y_,
             params_.resample_var_z_),
        Vec3(params_.resample_var_roll_,
             params_.resample_var_pitch_,
             params_.resample_var_yaw_)));
    pf_-> updateNoise(params_.odom_err_lin_lin_, params_.odom_err_lin_ang_,
                      params_.odom_err_ang_lin_, params_.odom_err_ang_ang_);
    publishParticles();

    auto biased_mean = pf_->biasedMean(odom_prev_, params_.num_particles_, params_.bias_var_dist_, params_.bias_var_ang_); 
    biased_mean.rot_.normalize();

    ROS_INFO("Flag5");
    assert(std::isfinit(biased_mean.pose_.x) && std::isfinit(biased_mean.pose_.y) && std::isfinit(biased_mean.pose_.z) &&
           std::isfinit(biased_mean.rot_.x) && std::isfinit(biased_mean.rot_.y) && std::isfinit(biased_mean.rot_.z) && std::isfinit(biased_mean.rot_.w));
    
    publishPose(biased_mean, header);  


    // 7. Publish map tf
    ros::Time localized_current = ros::Time::now();
    float dt = (localized_current - localized_last_).toSec();
    if (dt > 1.0) dt = 1.0;
    else if ( dt < 0.0) dt = 0.0;
    localized_last_ = localized_current;   

    ROS_INFO("Flag6");    
    Vec3 map_pose = biased_mean.pose_ - biased_mean.rot_.inv() * odom_.pose_;
    Quat map_rot = biased_mean.rot_ * odom_.rot_.inv();
    
    // broadcastTf(map_pose, map_rot);

}

void AMCL3DNode::broadcastTf(Vec3 map_pose, Quat map_rot) // map to odom
{
    geometry_msgs::TransformStamped trans;
    if (has_odom_)
        trans.header.stamp = odom_last_time_; //tf_tolerance_base_+ *params_.tf_tolerance_;
    else
        trans.header.stamp = ros::Time::now(); //tf_tolerance_base_+ *params_.tf_tolerance_;
    trans.header.frame_id = params_.frame_ids_["map"];
    trans.child_frame_id = params_.frame_ids_["odom"];
    
    trans.transform.translation = tf2::toMsg(tf2::Vector3(map_pose.x_, map_pose.y_, map_pose.z_));
    trans.transform.rotation = tf2::toMsg(tf2::Quaternion(map_rot.x_, map_rot.y_, map_rot.z_, map_rot.w_));

    std::vector<geometry_msgs::TransformStamped> transforms;
    transforms.push_back(trans);
    tfb_.sendTransform(transforms);
}

void AMCL3DNode::publishPose(PoseState& biased_mean, const std_msgs::Header& header)
{
    geometry_msgs::PoseWithCovarianceStamped pose;

    geometry_msgs::TransformStamped trans;
    trans.header.frame_id = params_.frame_ids_["map"];
    trans.child_frame_id = params_.frame_ids_["floor"];
    trans.transform.translation = tf2::toMsg(tf2::Vector3(0.0, 0.0, biased_mean.pose_.z_));
    trans.transform.rotation = tf2::toMsg(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));

    //transforms.push_back(trans);

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
    pub_pose_.publish(pose);
}

void AMCL3DNode::publishParticles()
{
    geometry_msgs::PoseArray particles;
    if (has_odom_)
        particles.header.stamp = odom_last_time_; // + tf_tolerance_base_ + *params_.tf_tolerance_;
    else
        particles.header.stamp = ros::Time::now();// + tf_tolerance_base_ + *params_.tf_tolerance_;
    particles.header.frame_id = params_.frame_ids_["map"];
    
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
    
    pub_particle_.publish(particles);        
}
       
} // namespace amcl3d
