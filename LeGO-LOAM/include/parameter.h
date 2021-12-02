#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <map>

#include <ros/ros.h>

#include <state/state_rep.h>

class Parameters
{
public:
    bool load(ros::NodeHandle& pnh)
    {
        ROS_INFO("parameter loading");

        pnh.param("pop_err", pop_err_, 0.01); 
        pnh.param("pop_z", pop_z_, 3.0); 

        pnh.param<std::string>("map_save_path", map_save_path_, "C:\\Users\\Haeyeon Kim\\Desktop\\lego_loam_result\\lego_loam_map.pcd"); 
        pnh.param("use_initial_pose", use_initial_pose_, false); 
        pnh.param("initial_x", initial_x_, 0.0); 
        pnh.param("initial_y", initial_y_, 0.0); 
        pnh.param("initial_yaw", initial_z_, 0.0); 
        pnh.param("descriptor_coeff", descriptor_coeff_, 0.0); 
        
        
        pnh.param("sampling_covariance", sampling_covariance_, 0.1); 
        pnh.param("rot_x", rot_x_, 0.0);
        pnh.param("rot_y", rot_y_, -1.570795);
        pnh.param("rot_z", rot_z_, -1.570795); // for lego loam
        pnh.param("map_roll", map_roll_, 1.570795);
        pnh.param("map_pitch", map_pitch_, 0.0);
        pnh.param("map_yaw", map_yaw_, 1.570795);

        pnh.param("max_particles", max_particles_, 100);
        pnh.param("min_particles", min_particles_, 20);
        pnh.param("map_frame", frame_ids_["map"], std::string("map"));
        pnh.param("base_frame", frame_ids_["base_link"], std::string("base_link"));
        pnh.param("odom_frame", frame_ids_["odom"], std::string("odom"));
        
        pnh.param("map_downsample_x", map_downsample_x_, 0.1);
        pnh.param("map_downsample_y", map_downsample_y_, 0.1);
        pnh.param("map_downsample_z", map_downsample_z_, 0.1);
        map_grid_min_ = std::min(std::min(map_downsample_x_, map_downsample_y_), map_downsample_z_);
        map_grid_max_ = std::max(std::max(map_downsample_x_, map_downsample_y_), map_downsample_z_);

        double map_update_interval_t;
        pnh.param("map_update_interval_interval", map_update_interval_t, 2.0);
        map_update_interval_.reset(new ros::Duration(map_update_interval_t));

        pnh.param("max_z_pose", max_z_pose_, 0.0);
        pnh.param("min_z_pose", min_z_pose_, 0.0);
        pnh.param("skip_measure", skip_measure_, 1);

        pnh.param("max_search_radius", max_search_radius_, 0.2);
        pnh.param("min_search_radius", min_search_radius_, 0.05);

        pnh.param("map_chunk", map_chunk_, 20.0);

        pnh.param("num_points", num_points_, 96);
        pnh.param("num_points_global", num_points_global_, 8);

        pnh.param("downsample_x", downsample_x_, 0.1);
        pnh.param("downsample_y", downsample_y_, 0.1);
        pnh.param("downsample_z", downsample_z_, 0.05);

        pnh.param("resample_var_x", resample_var_x_, 0.05);
        pnh.param("resample_var_y", resample_var_y_, 0.05);
        pnh.param("resample_var_z", resample_var_z_, 0.05);
        pnh.param("resample_var_roll", resample_var_roll_, 0.05);
        pnh.param("resample_var_pitch", resample_var_pitch_, 0.05);
        pnh.param("resample_var_yaw", resample_var_yaw_, 0.05);

        pnh.param("odom_lin_err_tc", odom_lin_err_tc_, 10.0);
        pnh.param("odom_lin_err_sigma", odom_lin_err_sigma_, 100.0);
        pnh.param("odom_ang_err_tc", odom_ang_err_tc_, 10.0);
        pnh.param("odom_ang_err_sigma", odom_ang_err_sigma_, 100.0);
        pnh.param("odom_err_lin_lin", odom_err_lin_lin_, 0.10);
        pnh.param("odom_err_lin_ang", odom_err_lin_ang_, 0.05);
        pnh.param("odom_err_ang_lin", odom_err_ang_lin_, 0.05);
        pnh.param("odom_err_ang_ang", odom_err_ang_ang_, 0.05);

        pnh.param("update_downsample_x", update_downsample_x_, 0.3);
        pnh.param("update_downsample_y", update_downsample_y_, 0.3);
        pnh.param("update_downsample_z", update_downsample_z_, 0.3);

        pnh.param("bias_var_dist", bias_var_dist_, 2.0);
        pnh.param("bias_var_ang", bias_var_ang_, 1.57);
        pnh.param("acc_var", acc_var_, M_PI / 4.0);  // 45 deg

        double clip_near, clip_far;
        pnh.param("clip_near", clip_near, 0.5);
        pnh.param("clip_far", clip_far, 4.0);
        clip_near_sq_ = clip_near * clip_near;
        clip_far_sq_ = clip_far * clip_far;

        double clip_z_min, clip_z_max;
        pnh.param("clip_z_min", clip_z_min, -2.0);
        pnh.param("clip_z_max", clip_z_max, 2.0);
        clip_z_min_ = clip_z_min;
        clip_z_max_ = clip_z_max;
        pnh.param("perform_weighting_ratio", perform_weighting_ratio_, 2.0);
        pnh.param("max_weight_ratio", max_weight_ratio_, 5.0);
        pnh.param("max_weight", max_weight_, 5.0);
        pnh.param("normal_search_range", normal_search_range_, 0.4);


        pnh.param("expansion_var_x", expansion_var_x_, 0.2);
        pnh.param("expansion_var_y", expansion_var_y_, 0.2);
        pnh.param("expansion_var_z", expansion_var_z_, 0.2);
        pnh.param("expansion_var_roll", expansion_var_roll_, 0.05);
        pnh.param("expansion_var_pitch", expansion_var_pitch_, 0.05);
        pnh.param("expansion_var_yaw", expansion_var_yaw_, 0.05);
        pnh.param("match_ratio_thresh", match_ratio_thresh_, 0.0);
        pnh.param("dist_tolerance", dist_tolerance_, 100.0);
        return true;
    }

public:
    std::map<std::string, std::string> frame_ids_;
    std::string map_save_path_;
    int max_particles_, min_particles_;
    int skip_measure_;
    
    double pop_err_, pop_z_;
    bool use_initial_pose_;
    double initial_x_, initial_y_, initial_z_;

    double rot_x_, rot_y_, rot_z_;
    double map_roll_, map_pitch_, map_yaw_;
    double sampling_covariance_;

    double dist_tolerance_;
    double descriptor_coeff_;
    double map_downsample_x_;
    double map_downsample_y_;
    double map_downsample_z_;
    double map_grid_min_;
    double map_grid_max_;
    double downsample_x_;
    double downsample_y_;
    double downsample_z_;

    double update_downsample_x_;
    double update_downsample_y_;
    double update_downsample_z_;

    double resample_var_x_;
    double resample_var_y_;
    double resample_var_z_;
    double resample_var_roll_;
    double resample_var_pitch_;
    double resample_var_yaw_;

    double odom_err_lin_lin_;
    double odom_err_lin_ang_;
    double odom_err_ang_lin_;
    double odom_err_ang_ang_;

    std::shared_ptr<ros::Duration> map_update_interval_;

    double odom_lin_err_tc_;
    double odom_ang_err_tc_;

    double odom_lin_err_sigma_;
    double odom_ang_err_sigma_;

    double map_chunk_;

    double bias_var_dist_;
    double bias_var_ang_;
    double acc_var_;
    int num_points_, num_points_global_; // for lidar measurement
    double max_search_radius_, min_search_radius_;

    double clip_near_sq_, clip_far_sq_;
    double clip_z_min_, clip_z_max_;

    double perform_weighting_ratio_;
    double max_weight_ratio_;
    double max_weight_;
    double normal_search_range_;

    double expansion_var_x_;
    double expansion_var_y_;
    double expansion_var_z_;
    double expansion_var_roll_;
    double expansion_var_pitch_;
    double expansion_var_yaw_;
    double match_ratio_thresh_;
    double max_z_pose_, min_z_pose_;
};
#endif