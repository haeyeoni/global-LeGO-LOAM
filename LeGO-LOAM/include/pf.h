#pragma once
#ifndef PF_H
#define PH_H

#include <algorithm>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

#include <state/state_rep.h>
#include <chunked_kdtree.h>
#include <gaussianMixture.h>
#include <normal_likelihood.h>
template <typename T> // T: state type
class Particle 
{
    
public:
    Particle() // initialize
    {
        probability_ = 0.0;
        probability_bias_ = 0.0; 
    }
    explicit Particle(float prob): accum_probability_(prob) {}

    T state_;

    float probability_;
    float probability_bias_;
    float accum_probability_;

    bool operator<(const Particle& p2) const
    {
        return this->accum_probability_ < p2.accum_probability_;
    }

};

template <typename T> // T: PoseState
class ParticleFilter
{
public:
    using Ptr = std::shared_ptr<ParticleFilter>;
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

    std::vector<Particle<T>> particles_;
    std::vector<Particle<T>> particles_dup_;
    std::random_device random_seed;
    std::default_random_engine engine_;

    GaussianMixture<PoseState>::Ptr gmm_;

    explicit ParticleFilter(const int num_particles, const double sampling_covariance)
    {
        particles_.resize(num_particles);
        gmm_.reset(new GaussianMixture<PoseState>(sampling_covariance));        
    }

    void init(std::vector<float> poses)
    {
        std::cout<<"initializing particle filter"<<std::endl;
        gmm_->setDistribution(poses);

        for (auto& p: particles_)
        {
            p.state_ = gmm_->sample();
            p.probability_ = 1.0 / particles_.size();
        }
        std::cout<<"finish initializing particle filter"<<std::endl;
    }

    T generateNoise(T sigma)
    {
        T noise;
        std::vector<float> org_noise(sigma.size());
        for (size_t i = 0; i < sigma.size(); i++)
        {
            std::normal_distribution<float> nd(0.0, sigma[i]);
            org_noise[i] = nd(engine_);
        }        

        // change to PoseState datatype
        for (size_t i = 0; i < 3; i ++) // x,y,z , linear odom error
            noise[i] = noise[i + 7] = org_noise[i];
        
        Vec3 rpy_noise;
        for (size_t i = 0; i < 3; i ++) 
        {
            rpy_noise[i] = org_noise[i + 3]; 
            noise[i + 10] = org_noise[i + 3]; 
        }
        noise.rot_ = Quat(rpy_noise);
        return noise;
    }

    T generateMultivariateNoise(T mean, Matrix covariance)
    {
        const size_t dim = 6; //x,y,z,roll,pitch,yaw
        
        T noise;

        Vector noise_vector(dim);
        Vector mean_vector(dim);

        Vec3 rpy_mean = mean.rot_.getRPY();
        
        const Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(covariance);
        Matrix norm_transform = eigen_solver.eigenvectors() * eigen_solver.eigenvalues().cwiseSqrt().asDiagonal(); // covariance transform
        std::normal_distribution<float> nd(0.0, 1.0);
        
        for (size_t i = 0; i < dim; i ++)
        {
            noise_vector[i] = nd(engine_);
            if (i < dim / 2)
                mean_vector[i] = mean[i];
            else
                mean_vector[i] = rpy_mean[i-3];
        }
        noise_vector = mean_vector + norm_transform * noise_vector;
        // // change to PoseState datatype
        for (size_t i = 0; i < 3; i ++) // x,y,z , linear odom error
            noise[i] = noise[i + 7] = noise_vector[i];
        
        Vec3 rpy_noise;
        for (size_t i = 0; i < 3; i ++) 
        {
            rpy_noise[i] = noise_vector[i + 3]; 
            noise[i + 10] = noise_vector[i + 3] - mean[i + 3]; 
        }
        noise.rot_ = Quat(rpy_noise);
        return noise;
    }

    size_t getParticleSize()
    {
        return particles_.size();
    }

    T getParticle(const size_t i) const
    {
        return particles_[i].state_;
    }

    T mean()
    {
        T mean_state;
        float p_sum = 0.0;

        for (auto& p : particles_)
        {
            p_sum += p.probability_;
            T temp = p.state_;
            for (size_t i = 0; i < temp.size(); i ++) // 1~13
                mean_state[i] += temp[i] * p.probability_;
        }    
        assert(p_sum > 0.0);
        mean_state = mean_state / p_sum;
            
        return mean_state;
    }

    T biasedMean(PoseState& prev_s, int num_particles, float bias_var_dist, float bias_var_ang)
    {
        float p_bias;
        if (particles_.size() < num_particles)
            for (auto& p : particles_)
                p.probability_bias_ = 1.0;
        else
        {
            NormalLikelihood nl_lin(bias_var_dist);
            NormalLikelihood nl_ang(bias_var_ang);
            for (auto& p : particles_)
            {
                const float lin_diff = (p.state_.pose_ - prev_s.pose_).norm(); // linear movement 
                Vec3 axis;
                float ang_diff;
                (p.state_.rot_ * prev_s.rot_.inv()).getAxisAng(axis, ang_diff); // angular movement                 
                p_bias = nl_lin(lin_diff) * nl_ang(ang_diff) + 1e-6;
                assert(std::isfinite(p_bias));
                p.probability_bias_ = p_bias;  
            }
        }

        T biased_mean_state = T();

        float p_sum = 0.0;

        Vec3 front_sum, up_sum;

        
        for (auto& p : particles_)
        {
            p_sum += p.probability_ * p.probability_bias_;

            T temp = p.state_;

            front_sum += temp.rot_ * Vec3(1.0, 0.0, 0.0) * p.probability_ * p.probability_bias_;
            up_sum += temp.rot_ * Vec3(0.0, 0.0, 1.0) * p.probability_ * p.probability_bias_;
            for (size_t i = 0; i < biased_mean_state.size(); i ++)
                biased_mean_state[i] += temp[i] * (p.probability_ * p.probability_bias_);
        }    
            
        assert(p_sum > 0.0);
        biased_mean_state = biased_mean_state / p_sum;
        biased_mean_state.rot_ = Quat(front_sum, up_sum);
        return biased_mean_state;
    }

    std::vector<T> covariance(const float random_sample_ratio = 1.0)
    {
        T m = mean();
        std::vector<T> cov;
        float sum = 0.0;
        float cov_element;
        cov.resize(m.size());
        
        size_t p_num = getParticleSize();
        float p_sum = 0.0;
        std::vector<size_t> indices(p_num);
        std::iota(indices.begin(), indices.end(), 0); // insert the number into indices from 0 ~ particle_num
        if (random_sample_ratio < 1.0)
        {
            std::shuffle(indices.begin(), indices.end(), engine_);
            const size_t sample_num = 
                    std::min(p_num, std::max(size_t(0), static_cast<size_t>(p_num * random_sample_ratio))); // decide the sample size
            indices.resize(sample_num);
        }

        for (size_t i : indices) // shuffled, resized indices
        {
            auto& p = particles_[i];
            p_sum += p.probability_;
            for (size_t j = 0; j < m.size(); j ++) // for all state
                for (size_t k = j; k < m.size(); k ++) 
                {
                    cov_element = (p.state_[k] - m[k]) * (p.state_[j] - m[j]);
                    cov[k][j] = cov[j][k] += cov_element * p.probability_;
                }
        }

        for (size_t j = 0; j < m.size(); j ++) // normalize the covariance 
            for (size_t k = 0; k < m.size(); k ++) 
                cov[k][j] /= p_sum;
        
        return cov;
    }

    T maxProbState()
    {
        T* max_prob_state = &particles_[0].state_;
        float max_probability = particles_[0].probability_;
        for (auto& p : particles_)
        {
            if (max_probability < p.probability_)
            {
                max_probability = p.probability_;
                max_prob_state = &p.state_;
            }
        }
        return *max_prob_state;
    }

    void resetParticleOdomErr()
    {
        for (auto& p : particles_)
        {
            p.state_.odom_lin_err_ = Vec3();
            p.state_.odom_ang_err_ = Vec3();
        }
    }

    void resample(T sigma)
    {
        float accum = 0; // for sampling

        for (auto& p : particles_)
        {
            accum += p.probability_;
            p.accum_probability_ = accum; 
        }

        particles_dup_ = particles_; 
        std::sort(particles_dup_.begin(), particles_dup_.end());

        const float step = accum / particles_.size(); 
        const float initial_p = std::uniform_real_distribution<float>(0.0, step)(engine_); //first particle (random) (0 < random < step)
        auto it = particles_dup_.begin(); // first particle
        auto it_prev = particles_dup_.begin(); //first particle 
        const float prob = 1.0 / particles_.size(); // uniform probability 

        for (size_t i = 0; i < particles_.size(); ++i)
        {
            auto& p = particles_[i];
            const float p_scan = step * i + initial_p;
            it = std::lower_bound(it, particles_dup_.end(), Particle<T> (p_scan)); // lower_bound(arr, arr+n, key)
            p.probability_ = prob; // reset the probability
            p.state_ = it->state_ + generateNoise(sigma);
            it_prev = it;
        }
    }

    void updateNoise(float noise_ll, float noise_la, float noise_al, float noise_aa)
    {
        std::normal_distribution<float> noise(0.0, 1.0);
        for (auto& p : particles_)
        {
            p.state_.noise_ll_ = noise(engine_) * noise_ll;
            p.state_.noise_la_ = noise(engine_) * noise_la;
            p.state_.noise_al_ = noise(engine_) * noise_al;
            p.state_.noise_aa_ = noise(engine_) * noise_aa;
        }
    }
};

#endif
