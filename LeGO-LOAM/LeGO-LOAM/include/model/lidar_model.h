#ifndef LIDAR_MODEL_H
#define LIDAR_MODEL_H

#include <state/state_rep.h>
#include <chunked_kdtree.h>
#include <pointcloud_sampler.h>
#include <pcl/filters/extract_indices.h>
#include <normal_likelihood.h>

typedef struct
{
    float likelihood;
    float match_ratio;
} lidar_measure_result_t;

class LidarModel 
{
public:
    using PointType = pcl::PointXYZ;
    using Ptr = std::shared_ptr<LidarModel>;
    PointCloudSampler<PointType>::Ptr sampler_;

private:
    size_t num_points_;
    size_t num_points_default_;
    size_t num_points_global_;
    float clip_far_sq_; // maximum square distance 
    float clip_near_sq_; // minimum square distance
    float clip_z_min_;
    float clip_z_max_;
    float match_weight_;
    float max_search_radius_; 
    float min_search_radius_; 
    ChunkedKdtree<PointType>::Ptr kdtree_;
    const float odom_lin_err_sigma_;

public:
    LidarModel(int num_points, int num_points_global, double max_search_radius, double min_search_radius, 
               double clip_near_sq, double clip_far_sq, double clip_z_min, double clip_z_max,
               double perform_weighting_ratio, double max_weight_ratio, 
               double max_weight, double normal_search_range,
               float odom_lin_err_sigma)
    : num_points_(num_points), num_points_global_(num_points_global), 
      max_search_radius_(max_search_radius), min_search_radius_(min_search_radius),
      clip_near_sq_(clip_near_sq), clip_far_sq_(clip_far_sq),
      clip_z_min_(clip_z_min), clip_z_max_(clip_z_max),
      odom_lin_err_sigma_(odom_lin_err_sigma)
    {
        num_points_default_ = num_points_ = num_points;
        sampler_.reset(new PointCloudSampler<PointType>(perform_weighting_ratio, max_weight_ratio, max_weight, normal_search_range));   
        match_weight_ = 5.0; 
    }

    void setParticleNumber(const size_t num_particles, const size_t current_num_particles)
    {
        if (current_num_particles <= num_particles)
        {
            num_points_ = num_points_default_;
            return;
        }
        size_t num = num_points_default_ * num_particles / current_num_particles;
        if (num < num_points_global_)
            num = num_points_global_;
        num_points_ = num;    
    }

    void setKdtree(ChunkedKdtree<PointType>::Ptr& kdtree)
    {
        kdtree_ = kdtree;
    }
    
    pcl::PointCloud<PointType>::Ptr filter(const pcl::PointCloud<PointType>::Ptr& pc) const
    {
        ROS_DEBUG("lidar filter called");
        pcl::PointCloud<PointType>::Ptr pc_filtered(new pcl::PointCloud<PointType>);
        pcl::PointIndices::Ptr outliers(new pcl::PointIndices());
        pcl::ExtractIndices<PointType> extract;
        *pc_filtered = *pc; // initial input
        
        for (int i = 0; i < (*pc_filtered).size(); i ++)
        {
            PointType p(pc_filtered->points[i].x, pc_filtered->points[i].y, pc_filtered->points[i].z);
            
            if ((p.x * p.x + p.y * p.y) > clip_far_sq_ || (p.x * p.x + p.y * p.y) < clip_near_sq_
                || p.z < clip_z_min_ || p.z > clip_z_max_ )
                outliers->indices.push_back(i);
        }
        extract.setInputCloud(pc_filtered);
        extract.setIndices(outliers);
        extract.setNegative(true);
        extract.filter(*pc_filtered);

        pc_filtered->width = 1;
        pc_filtered->height = pc_filtered->points.size();
        
        return sampler_->normal_sample(pc_filtered, num_points_); // sample num_points number of points from pc_filtered     
    }

    PointCloudSampler<PointType>::Ptr getSampler()
    {   
        return sampler_;
    }

    float measureCorrect(ParticleFilter<PoseState>::Ptr pf, pcl::PointCloud<PointType>::Ptr& pc_locals, const std::vector<Vec3>& origins)
    {
        // update particle probability 
    
        NormalLikelihood odom_lin_error_nl(odom_lin_err_sigma_); // generate normal likelihood 
        float match_ratio_max = 0.0;
        float likelihood_score, match_ratio, odom_error_likelihood;
        float sum = 0;
    
        for (auto& particle : pf->particles_)
        {
            if (!pc_locals || pc_locals->size() == 0)
            {
                likelihood_score = 1;
                match_ratio = 0;   
            }
            else
            {
                pcl::PointCloud<PointType>::Ptr pc_particle(new pcl::PointCloud<PointType>);
                
                std::vector<int> id(1);
                std::vector<float> sqr_distance(1);
                likelihood_score = 0;
                *pc_particle = *pc_locals; // initialize
                particle.state_.transform(*pc_particle); // transform pointcloud according to the particle's state
                
                size_t num = 0;
                if(pc_particle->is_dense)
                {                
                    for (const PointType& point : pc_particle->points) // for all point cloud points
                    {
                        if (kdtree_->radiusSearch(point, max_search_radius_, id, sqr_distance, 1)) // find nearest neighbor within (search_radius_) radius from map and if neighbor exists 
                        {
                            const float dist = max_search_radius_ - std::max(std::sqrt(sqr_distance[0]), min_search_radius_); // if the distance is smaller, likelihood is higher 
                            if (dist < 0.0)
                                continue;
                            likelihood_score += dist * match_weight_; 
                            num ++;                    
                        }
                    }
                }
                const float match_ratio = static_cast<float>(num) / pc_particle->points.size();
                if(match_ratio_max < match_ratio)
                    match_ratio_max = match_ratio;
            }
            odom_error_likelihood = odom_lin_error_nl(particle.state_.odom_lin_err_.norm());
            particle.probability_ *= likelihood_score * odom_error_likelihood;
            
            sum += particle.probability_;
        }

        for (auto& particle : pf->particles_)
        {
            particle.probability_ /= sum;
        }    
        return match_ratio_max;
    }

};


#endif