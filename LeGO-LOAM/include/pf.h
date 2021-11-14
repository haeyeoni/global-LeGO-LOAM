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
#include <pf_kdtree.h>
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
typedef struct
{
    int count;
    double weight;
    T meanPose;
    double m[5], c[2][2];
    double cov[3][3];
} pf_cluster_t;

public:
    using Ptr = std::shared_ptr<ParticleFilter>;
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

    std::vector<Particle<T>> particles_;
    std::vector<Particle<T>> particles_dup_;
    std::vector<int> limit_cache;
    T pf_mean_pose_;
    std::random_device random_seed;
    std::default_random_engine engine_;
    std::vector<pf_cluster_t> clusters;
    pf_kdtree_t *kdtree;
    int cluster_max_count;
    int num_particles_, max_particles_, min_particles_;
    double m[5], c[2][2];
    double cov[3][3];
    double pop_err_, pop_z_;

    GaussianMixture<PoseState>::Ptr gmm_;

    explicit ParticleFilter(const int max_particles, const int min_particles, 
                        const double sampling_covariance, const double pop_err, const double pop_z)
    {
        particles_.resize(max_particles);
        particles_dup_.resize(max_particles);
        gmm_.reset(new GaussianMixture<PoseState>(sampling_covariance)); 
        kdtree = pf_kdtree_alloc(3 * max_particles);
        max_particles_ = max_particles; 
        min_particles_ = min_particles;   
        num_particles_ = max_particles;  

        limit_cache.reserve(max_particles);
        cluster_max_count = max_particles;
        clusters.reserve(max_particles);
        pop_err_ = pop_err;
        pop_z_ = pop_z_;
    }

    void init(std::vector<float> poses)
    {
        printf("initializing particle filter (particle number: %d) \n", num_particles_);
        
        pf_kdtree_clear(kdtree);

        gmm_->setDistribution(poses, 1);
        Particle<T> *p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            p->state_ = gmm_->sample();
            p->probability_ = 1.0 / num_particles_;
            pf_kdtree_insert(kdtree, p->state_, 1.0 / num_particles_);
        }

        pf_cluster_stats();
        printf("finishing inserting to kdtree (node: %d) \n", kdtree->leaf_count);
    }

    void reinit(std::vector<float> poses, float distance)
    {
        T mean_pose = mean();
        
        pf_kdtree_clear(kdtree);

        poses.push_back(mean_pose[0]);
        poses.push_back(mean_pose[1]);
        poses.push_back(mean_pose[2]);
        
        gmm_->setDistribution(poses, 2);
        Particle<T> *p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            p->state_ = gmm_->sample();
            p->probability_ = 1.0 / num_particles_;
            pf_kdtree_insert(kdtree, p->state_, 1.0 / num_particles_);
        }
        pf_cluster_stats();
    }

    void pf_cluster_stats()
    {
        int i, j, k, cidx;
        // Workspace
        size_t count;
        double weight;
        // Cluster the samples
        pf_kdtree_cluster(kdtree);
        // Initialize cluster stats
        int cluster_count = 0;

        pf_cluster_t *cluster;

        for (i = 0; i < cluster_max_count; i++)
        {
            pf_cluster_t cluster_new;
            cluster_new.count = 0;
            cluster_new.weight = 0;
            cluster_new.meanPose = T();

            for (j = 0; j < 5; j++)
                cluster_new.m[j] = 0.0;

            for (j = 0; j < 2; j++)
                for (k = 0; k < 2; k++)
                    cluster_new.c[j][k] = 0.0;
            clusters[i] = cluster_new;
            cluster_count ++;
        }

        // Initialize overall filter stats
        count = 0;
        weight = 0.0;
        pf_mean_pose_ = T();

        // Compute cluster stats
        Particle<T> p;
        for (i = 0; i < num_particles_; i ++)
        {
            p = particles_[i];
           // Get the cluster label for this sample
            cidx = pf_kdtree_get_cluster(kdtree, p.state_);
            
            assert(cidx >= 0);
            if (cidx >= cluster_max_count)
                continue;
            if (cidx + 1 > cluster_count)
                cluster_count = cidx + 1;
            
            cluster = & clusters[cidx];

            cluster->count += 1;
            cluster->weight += p.probability_;

            count += 1;
            weight += p.probability_;

            // // Compute mean
            cluster->m[0] += p.probability_ * p.state_.pose_.x_; // sample->pose.v[0];
            cluster->m[1] += p.probability_ * p.state_.pose_.y_; //sample->pose.v[1];
            cluster->m[2] += p.probability_ * cos(p.state_.rot_.getRPY().z_); // cos(sample->pose.v[2]);
            cluster->m[3] += p.probability_ * sin(p.state_.rot_.getRPY().z_); // sin(sample->pose.v[2]);
            cluster->m[4] += p.probability_ * p.state_.pose_.z_; //sample->pose.v[2];

            m[0] += p.probability_ * p.state_.pose_.x_; // sample->pose.v[0];
            m[1] += p.probability_ * p.state_.pose_.y_; //sample->pose.v[1];
            m[2] += p.probability_ * cos(p.state_.rot_.getRPY().z_); // cos(sample->pose.v[2]);
            m[3] += p.probability_ * sin(p.state_.rot_.getRPY().z_); // cos(sample->pose.v[2]);
            m[4] += p.probability_ * p.state_.pose_.z_; //sample->pose.v[2];
            
    

            cluster->c[0][0] += p.probability_ * p.state_.pose_.x_ * p.state_.pose_.x_;
            c[0][0] += p.probability_ * p.state_.pose_.x_ * p.state_.pose_.x_;

            cluster->c[0][1] += p.probability_ * p.state_.pose_.x_ * p.state_.pose_.x_;
            c[0][1] += p.probability_ * p.state_.pose_.y_ * p.state_.pose_.x_;

            cluster->c[1][0] += p.probability_ * p.state_.pose_.y_ * p.state_.pose_.x_;
            c[1][0] += p.probability_ * p.state_.pose_.y_ * p.state_.pose_.x_;

            cluster->c[1][1] += p.probability_ * p.state_.pose_.y_ * p.state_.pose_.y_;
            c[1][1] += p.probability_ * p.state_.pose_.y_ * p.state_.pose_.y_;
        }

        // Normalize
        for (i = 0; i < cluster_count; i++)
        {
            cluster = & clusters[i];

            cluster->meanPose = T(Vec3(cluster->m[0] / cluster->weight, cluster->m[1] / cluster->weight, cluster->m[4] / cluster->weight), Vec3(0, 0, atan2(cluster->m[3], cluster->m[2])));
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    cluster->cov[j][k] = 0.0;
            // Covariance in linear components
            cluster->cov[0][0] = cluster->c[0][0] / cluster->weight - cluster->meanPose.pose_.x_ * cluster->meanPose.pose_.x_;
            cluster->cov[0][1] = cluster->c[0][1] / cluster->weight - cluster->meanPose.pose_.x_ * cluster->meanPose.pose_.y_;
            cluster->cov[1][0] = cluster->c[1][0] / cluster->weight - cluster->meanPose.pose_.x_ * cluster->meanPose.pose_.y_;
            cluster->cov[1][1] = cluster->c[1][1] / cluster->weight - cluster->meanPose.pose_.y_ * cluster->meanPose.pose_.y_;


            // Covariance in angular components; I think this is the correct
            // formula for circular statistics.
            cluster->cov[2][2] = -2 * log(sqrt(cluster->m[2] * cluster->m[2] +
                                                cluster->m[3] * cluster->m[3]));

        }

        assert(fabs(weight) >= DBL_EPSILON);
        if (fabs(weight) < DBL_EPSILON)
        {
            printf("ERROR : divide-by-zero exception : weight is zero\n");
            return;
        }
        // Compute overall filter stats
        pf_mean_pose_ = T(Vec3(m[0] / weight, m[1] / weight, m[4] / weight), Vec3(0,0,atan2(m[3], m[2])));
        // Covariance in linear components
        cov[0][0] = c[0][0] / weight - pf_mean_pose_.pose_.x_ * pf_mean_pose_.pose_.x_;
        cov[0][1] = c[0][1] / weight - pf_mean_pose_.pose_.x_ * pf_mean_pose_.pose_.y_;
        cov[1][0] = c[1][0] / weight - pf_mean_pose_.pose_.x_ * pf_mean_pose_.pose_.y_;
        cov[1][1] = c[1][1] / weight - pf_mean_pose_.pose_.y_ * pf_mean_pose_.pose_.y_;

        // Covariance in angular components; I think this is the correct
        // formula for circular statistics.
        cov[2][2] = -2 * log(sqrt(m[2] * m[2] + m[3] * m[3]));

        return;
    }


    T generateNoise(T sigma)
    {
        T noise;
        std::vector<float> org_noise(sigma.size());
        for (int i = 0; i < sigma.size(); i++)
        {
            std::normal_distribution<float> nd(0.0, sigma[i]);
            org_noise[i] = nd(engine_);
        }        

        // change to PoseState datatype
        for (int i = 0; i < 3; i ++) // x,y,z , linear odom error
            noise[i] = noise[i + 7] = org_noise[i];
        
        Vec3 rpy_noise;
        for (int i = 0; i < 3; i ++) 
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
        return num_particles_;
    }

    T getParticle(const size_t i) const
    {
        return particles_[i].state_;
    }

    T mean()
    {
        T mean_state;
        float p_sum = 0.0;
        Particle<T> p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = particles_[i];
            p_sum += p.probability_;
            T temp = p.state_;
            for (int j = 0; j < temp.size(); j ++) // 1~13
                mean_state[j] += temp[j] * p.probability_;
        }    
        assert(p_sum > 0.0);
        mean_state = mean_state / p_sum;
            
        return mean_state;
    }

    T biasedMean(T& prev_s, float bias_var_dist, float bias_var_ang)
    {
        // 1. calculate main clusters
        float p_bias;
        Particle<T> *p;
        // if (num_particles_ < num_particles)
        // {
        //     for (int i = 0; i < num_particles_; i ++)
        //     {
        //         p = &particles_[i];
        //         p->probability_bias_ = 1.0;
        //     }
        // }
        // else
        // {
        NormalLikelihood nl_lin(bias_var_dist);
        NormalLikelihood nl_ang(bias_var_ang);
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            const float lin_diff = (p->state_.pose_ - prev_s.pose_).norm(); // linear movement 
            Vec3 axis;
            float ang_diff;
            (p->state_.rot_ * prev_s.rot_.inv()).getAxisAng(axis, ang_diff); // angular movement                 
            p_bias = nl_lin(lin_diff) * nl_ang(ang_diff) + 1e-6;
            assert(std::isfinite(p_bias));
            p->probability_bias_ = p_bias;  
        }
        // }

        T biased_mean_state = T();

        float p_sum = 0.0;

        Vec3 front_sum, up_sum;

        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            p_sum += p->probability_ * p->probability_bias_;

            T temp = p->state_;

            front_sum += temp.rot_ * Vec3(1.0, 0.0, 0.0) * p->probability_ * p->probability_bias_;
            up_sum += temp.rot_ * Vec3(0.0, 0.0, 1.0) * p->probability_ * p->probability_bias_;
            for (size_t i = 0; i < biased_mean_state.size(); i ++)
                biased_mean_state[i] += temp[i] * (p->probability_ * p->probability_bias_);
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
        Particle<T> p;
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
        Particle<T> p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = particles_[i];
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
        Particle<T> *p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            p->state_.odom_lin_err_ = Vec3();
            p->state_.odom_ang_err_ = Vec3();
        }
    }

    void resample(T sigma)
    {
        float accum = 0; // for sampling
        double total = 0;
        int sample_count = 0;
        pf_kdtree_clear(kdtree);
        Particle<T> *p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            accum += p->probability_;
            p->accum_probability_ = accum; 
        }
        //copy particles
        std::copy( particles_.begin(), particles_.end(), particles_dup_.begin() );
        // particles_dup_ = particles_;  

        std::sort(particles_dup_.begin(), particles_dup_.end());
        const float step = accum / particles_.size(); 
        const float initial_p = std::uniform_real_distribution<float>(0.0, step)(engine_); //first particle (random) (0 < random < step)
        auto it = particles_dup_.begin(); // first particle
        auto it_prev = particles_dup_.begin(); //first particle  
        for (int i = 0; i < max_particles_; ++i)
        {
            sample_count ++;
            p = &particles_[i];
            const float p_scan = step * i + initial_p;
            it = std::lower_bound(it, particles_dup_.end(), Particle<T> (p_scan)); // lower_bound(arr, arr+n, key)
            p->probability_ = 1.0; // reset the probability
            p->state_ = it->state_ + generateNoise(sigma);
            it_prev = it;

            total += p->probability_;

            // // Add sample to histogram
            pf_kdtree_insert(kdtree, p->state_ , p->probability_);

            // // See if we have enough samples yet
            if (sample_count > pf_resample_limit(kdtree->leaf_count))
                break;
        }
        // Normalize weights
        for (int i = 0; i < sample_count; i++)
        {
            p = &particles_[i];
            p->probability_ /= total;
        }
        num_particles_ = sample_count;

        // Re-compute cluster statistics
        pf_cluster_stats();
    }
// Compute the required number of samples, given that there are k bins
// with samples in them.  This is taken directly from Fox et al.
    int pf_resample_limit(int k)
    {
        double a, b, c, x;
        int n;

        // Return max_samples in case k is outside expected range, this shouldn't
        // happen, but is added to prevent any runtime errors
        if (k < 1 || k > max_particles_)
            return max_particles_;

        // Return value if cache is valid, which means value is non-zero positive
        if (limit_cache[k-1] > 0)
            return limit_cache[k-1];

        if (k == 1)
        {
            limit_cache[k-1] = max_particles_;
            return max_particles_;
        }

        a = 1;
        b = 2 / (9 * ((double) k - 1));
        c = sqrt(2 / (9 * ((double) k - 1))) * pop_z_;
        x = a - b + c;

        n = (int) ceil((k - 1) / (2 * pop_err_) * x * x * x);

        if (n < min_particles_)
        {
            limit_cache[k-1] = min_particles_;
            return min_particles_;
        }
        if (n > max_particles_)
        {
            limit_cache[k-1] = max_particles_;
            return max_particles_;
        }
        
        limit_cache[k-1] = n;
        return n;
    }

    void updateNoise(float noise_ll, float noise_la, float noise_al, float noise_aa)
    {
        std::normal_distribution<float> noise(0.0, 1.0);
        Particle<T> *p;
        for (int i = 0; i < num_particles_; i ++)
        {
            p = &particles_[i];
            p->state_.noise_ll_ = noise(engine_) * noise_ll;
            p->state_.noise_la_ = noise(engine_) * noise_la;
            p->state_.noise_al_ = noise(engine_) * noise_al;
            p->state_.noise_aa_ = noise(engine_) * noise_aa;
        }
    }
};

#endif
