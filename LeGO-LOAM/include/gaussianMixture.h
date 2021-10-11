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
    std::vector<T> mean_poses_;
    double init_yaw_;

public:

    using Ptr = std::shared_ptr<GaussianMixture>;
    GaussianMixture(float sampling_covariance):sigma_(sampling_covariance), engine_(new std::default_random_engine(seed_gen_()))
    {}

    void setDistribution(int num_dist, float similarity_sum, std::vector<float> mean_poses, double init_yaw)
    {
        float acc_pi = 0.0;
        init_yaw_ = init_yaw;
        for (size_t i = 0; i < num_dist; i ++)
        {
            T mean_pose(Vec3(mean_poses[3*i + 0], mean_poses[3*i + 1], 0.0), Vec3(0.0, 0.0, 0.0));
            mean_poses_.push_back(mean_pose);
            pis_.push_back(mean_poses[3*i + 2] / similarity_sum);
            acc_pi += mean_poses[3*i + 2] / similarity_sum;
            cumulative_pi_.push_back(acc_pi);
        }
    }

    T sample()
    {
        // select normal distribution
        float random_value = ud(*engine_);
        auto it = std::lower_bound(cumulative_pi_.begin(), cumulative_pi_.end(), random_value); // return first larger than random_value in cumulative_weight
        
        const size_t n = it - cumulative_pi_.begin();
        std::normal_distribution<> x_distribution {mean_poses_[n][0], sigma_}; 
        std::normal_distribution<> y_distribution {mean_poses_[n][1], sigma_};
        
        T sample_pose (Vec3(x_distribution(*engine_), y_distribution(*engine_), 0.0), (Vec3(0.0 ,0.0 , (init_yaw_ + ud(*engine_)-0.5)*M_PI*2)));
        return sample_pose;
    }

};
#endif