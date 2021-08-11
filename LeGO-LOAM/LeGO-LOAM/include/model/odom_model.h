#ifndef ODOM_MODEL_H
#define ODOM_MODEL_H

#include <state/state_rep.h>
#include <pf.h>


class OdomModel 
{
public:
    using Ptr = std::shared_ptr<OdomModel>;

    OdomModel(float odom_lin_err_sigma, float odom_ang_err_sigma, float odom_lin_err_tc, float odom_ang_err_tc)
    : odom_lin_err_sigma_(odom_lin_err_sigma), odom_ang_err_sigma_(odom_ang_err_sigma),
      odom_lin_err_tc_(odom_lin_err_tc), odom_ang_err_tc_(odom_ang_err_tc)
    {
        std::cout<< "reset odom models"<<std::endl;
    }
    
    void setOdoms(const PoseState& odom_prev, const PoseState& odom_current, const float time_diff)
    {
        relative_translation_ = odom_prev.rot_.inv() * (odom_current.pose_ - odom_prev.pose_);
        relative_quat_ = odom_prev.rot_.inv() * odom_current.rot_;
        Vec3 axis;
        relative_quat_.getAxisAng(axis, relative_angle_); // calculate the axis and relative angle with the quaternion data
        relative_translation_norm_ = relative_translation_.norm();
        time_diff_ = time_diff;
    }

    void motionPredict(ParticleFilter<PoseState>::Ptr pf)
    {
        for (auto& p : pf->particles_)
        {
            // Position & angle update and noise calcuation     
            const Vec3 diff = relative_translation_ * (1.0 + p.state_.noise_ll_) + Vec3(p.state_.noise_al_ * relative_angle_, 0.0, 0.0); // linear translation with noise considered
            p.state_.odom_lin_err_ += (diff - relative_translation_); // only noise: rel_translation * noise_ll + relative_angle * noise_al
            
            p.state_.pose_ += p.state_.rot_ * diff;

            const float yaw_diff = p.state_.noise_la_ * relative_translation_norm_ + p.state_.noise_aa_ * relative_angle_; // angular translation noise in z-axis
            p.state_.rot_ = Quat(Vec3(0.0, 0.0, 1.0), yaw_diff) * p.state_.rot_ * relative_quat_;
            p.state_.rot_.normalize();                                                                     
            p.state_.odom_ang_err_ += Vec3(0.0, 0.0, yaw_diff);            
            p.state_.odom_lin_err_ *= (1.0 - time_diff_ / odom_lin_err_tc_); // if time_diff == tc -> error = 0 
            p.state_.odom_ang_err_ *= (1.0 - time_diff_ / odom_ang_err_tc_);
        }
    }

    ~OdomModel(){};

    private:
        Vec3 relative_translation_;
        Quat relative_quat_;
        float relative_angle_;
        float time_diff_;
        float relative_translation_norm_;
        const float odom_lin_err_sigma_;
        const float odom_ang_err_sigma_;
        const float odom_lin_err_tc_; // Time constant to forget the integral of the translational odometry error
        const float odom_ang_err_tc_; // Time constant to forget the integral of the rotational odometry error
};

#endif