#ifndef POINTCLOUD_SAMPLER_H
#define POINTCLOUD_SAMPLER_H

#include <memory>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

#include <state/state_rep.h>

template <class PointType>
class PointCloudSampler 
{
private: 
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    
    std::random_device seed_gen_;
    std::shared_ptr<std::default_random_engine> engine_;
    PoseState mean_;
    Matrix eigen_vectors_;
    Vector eigen_values_;
    double perform_weighting_ratio_;
    double max_weight_ratio_;
    double max_weight_;
    double normal_search_range_;

public:

    using Ptr = std::shared_ptr<PointCloudSampler>;
    PointCloudSampler(double perform_weighting_ratio, double max_weight_ratio, 
                       double max_weight, double normal_search_range)
        : perform_weighting_ratio_(perform_weighting_ratio),
      max_weight_ratio_(max_weight_ratio),
      max_weight_(max_weight),
      normal_search_range_(normal_search_range),
      engine_(new std::default_random_engine(seed_gen_()))
     {std::cout<< "reset sampler models"<<std::endl;}

    void setParameters(const double perform_weighting_ratio, const double max_weight_ratio, 
                       const double max_weight, const double normal_search_range)
    {
        perform_weighting_ratio_ = perform_weighting_ratio;
        max_weight_ratio_ = max_weight_ratio;
        max_weight_ = max_weight;
        normal_search_range_ = normal_search_range;
    }
    
    typename pcl::PointCloud<PointType>::Ptr uniform_sample(const typename pcl::PointCloud<PointType>::ConstPtr& pc, const size_t num) const
    {
        typename pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>);
        output->header = pc->header;

        if ((pc->points.size() == 0 )) return output;
        output->points.reserve(num);
        std::uniform_int_distribution<size_t> ud(0, pc->points.size() - 1);
        for (size_t i = 0; i < num; i ++)
            output->push_back(pc->points[ud(*engine_)]);
        return output;
    }

    typename pcl::PointCloud<PointType>::Ptr normal_sample(const typename pcl::PointCloud<PointType>::ConstPtr& pc, const size_t num) const
    {
        typename pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>);
        output->header = pc->header;
        
        if ((pc->points.size() == 0 ) || (num == 0))
            return output;
        
        if (pc->size() <= num)
        {
            *output = *pc;
            return output;
        }

        const double eigen_value_ratio = std::sqrt(eigen_values_[2] / eigen_values_[1]);
        double max_weight = 1.0;
        if (eigen_value_ratio < perform_weighting_ratio_)
            max_weight = 1.0;
        else if (eigen_value_ratio > max_weight_ratio_)
            max_weight = max_weight_;
        else
        {
            const double weight_ratio = (eigen_value_ratio - perform_weighting_ratio_) / (max_weight_ratio_ - perform_weighting_ratio_);
            max_weight = 1.0 + (max_weight_ - 1.0) * weight_ratio;
        }

        const Vec3 fpc_global(eigen_vectors_(0, 2), eigen_vectors_(1, 2), eigen_vectors_(2, 2)); // eigenvectors of covariance
        const Vec3 fpc_local = mean_.rot_.inv() * fpc_global;
        pcl::NormalEstimation<PointType, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        typename pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        
        ne.setInputCloud(pc);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(normal_search_range_);
        ne.compute(*cloud_normals); 

        std::vector<double> cumulative_weight(cloud_normals->points.size(), 0.0);
        for (size_t i = 0; i < cloud_normals->points.size(); i ++)
        {
            double weight = 1.0;
            const auto& normal = cloud_normals->points[i];
            if (!std::isnan(normal.normal_x) && !std::isnan(normal.normal_y) && !std::isnan(normal.normal_z))
            {
                double acos_angle = std::abs(normal.normal_x * fpc_local.x_ +
                                             normal.normal_y * fpc_local.y_ +
                                             normal.normal_z * fpc_local.z_);
                
                const double angle = std::acos(std::min(1.0, acos_angle)); // clipping acos_angle to 1.0 
                weight = 1.0 + (max_weight - 1.0) * ((M_PI / 2 - angle) / (M_PI / 2)); // ???
            }
            cumulative_weight[i] = weight + ((i == 0) ? 0.0 : cumulative_weight[i - 1]);
        }
        std::uniform_real_distribution<double> ud(0, cumulative_weight.back());
        std::unordered_set<size_t> selected_ids;
        while (true)
        {
            const double random_value = ud(*engine_);
            auto it = std::lower_bound(cumulative_weight.begin(), cumulative_weight.end(), random_value); // return first larger than random_value in cumulative_weight
            const size_t n = it - cumulative_weight.begin();
            selected_ids.insert(n);
            if(selected_ids.size() >= num)
                break;
        }
        
        output->points.reserve(num); // resize
        for (const auto& id : selected_ids)
            output->push_back(pc->points[id]);
        return output;
    }

    void setParticleStatistics(const PoseState& mean, const std::vector<PoseState>& covariances)
    {
        mean_ = mean;
        Matrix pose_cov(3, 3);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                pose_cov(i, j) = std::abs(covariances[i][j]);
    
        const Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(pose_cov);
        eigen_vectors_ = eigen_solver.eigenvectors();
        eigen_values_ = eigen_solver.eigenvalues();
    }

};

#endif
