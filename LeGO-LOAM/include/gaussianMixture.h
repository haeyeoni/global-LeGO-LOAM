#ifndef GMM_H
#define GMM_H

#include <algorithm>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <state/state_rep.h>

template <typename T>
class GaussianMixture
{
private:
    double *weight;
    double **mean;
    double sigma_;

    std::random_device seed_gen_;
    std::shared_ptr<std::default_random_engine> engine_;
    std::uniform_real_distribution<> ud{0.0, 1.0};     

protected:
    std::vector<float> pis_;
    std::vector<float> cumulative_pi_;
    T mean_pose_;
    double init_yaw_;

public:

    using Ptr = std::shared_ptr<GaussianMixture>;
    GaussianMixture(float sampling_covariance):sigma_(sampling_covariance), engine_(new std::default_random_engine(seed_gen_()))
    {}

    void setDistribution(std::vector<float> mean_pose)
    {
        mean_pose_ = T(Vec3(mean_pose[0], mean_pose[1], mean_pose[2]), Vec3(0.0, 0.0, 0.0));
        std::cout<<"mean pose: " <<mean_pose[0] <<" "<< mean_pose[1]<<mean_pose[2]<<std::endl;
    }

    T sample()
    {
        // select normal distribution
        float random_value = ud(*engine_);
        std::normal_distribution<> x_distribution {mean_pose_[0], sigma_}; 
        std::normal_distribution<> y_distribution {mean_pose_[1], sigma_}; 
        std::normal_distribution<> z_distribution {mean_pose_[2], sigma_};
        T sample_pose (Vec3(x_distribution(*engine_), y_distribution(*engine_), z_distribution(*engine_)), (Vec3(0.0 ,0.0 , (ud(*engine_)-0.5)*M_PI*2)));
        return sample_pose;
    }

};
#endif