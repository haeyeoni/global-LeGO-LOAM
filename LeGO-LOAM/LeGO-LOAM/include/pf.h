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

    void init(int length, float similaritySum, std::vector<float> poses)
    {
        std::cout<<"initializing particle filter"<<std::endl;
        gmm_->setDistribution(length, similaritySum, poses);

        for (auto& p: particles_)
        {
            p.state_ = gmm_->sample();
            p.probability_ = 1.0 / particles_.size();
        }
        std::cout<<"finish initializing particle filter"<<std::endl;
        
    }

    size_t getParticleSize()
    {
        return particles_.size();
    }

    T getParticle(const size_t i) const
    {
        return particles_[i].state_;
    }

};

#endif
