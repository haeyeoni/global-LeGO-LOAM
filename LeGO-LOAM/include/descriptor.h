#pragma once
#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>


#include <torch/torch.h>
#include <torch/script.h> 

#include "utility.h"
#include "feature_kdtree.h"


using PointType = pcl::PointXYZI;
const int featureSize = 64;

namespace lego_loam
{

class LocNetManager
{

template <typename T>
void templated_fn(T) {}
 
private:

    torch::jit::script::Module descriptor;
    std::ofstream featureFile;
    std::string featurePath;
    std::shared_ptr<kdtree<float, featureSize>> tree;

public: 
    LocNetManager(std::string path, bool mapping) 
    {
        featurePath = path;
        if (mapping)
            featureFile.open(path);        
    }
    ~LocNetManager()
    {
        featureFile.close();
    }

    float calculateDiff(at::Tensor currFeat, at::Tensor prevFeat)
    {
        float dist = 0.0;
        for(int i = 0; i < featureSize; i ++)
            dist += pow( currFeat[0][i].item<float>() - prevFeat[0][i].item<float>(), 2);
        
        return sqrt(dist);
    }

    /// Mapping
    void loadModel(string modelPath)
    {
        std::cout<< "reading from:" << modelPath <<std::endl;
        try {
            descriptor = torch::jit::load(modelPath);
        }
        catch (const c10::Error& e) {
            ROS_ERROR("error loading the model\n: %s", e);
            return;	    
        }
        ROS_INFO("Success loading model");    
    }

    at::Tensor makeDescriptor(const pcl::PointCloud<PointType>::Ptr laserCloudIn)
    {
        float verticalAngle, horizonAngle, range, prev_range, del_range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;


        int min_dist = 1;
        int max_dist = 81;

        int b = 80;
        int distBucket, bucket, iCount;
        
        cv::Mat imageCount = cv::Mat(N_SCAN, b, CV_32S, cv::Scalar::all(0));
        cv::Mat imageOne = cv::Mat(N_SCAN, b, CV_32F, cv::Scalar::all(0));
        
        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
    
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            distBucket = (max_dist - min_dist) / b;

            if (range > min_dist && range < max_dist)
            {
                for (int i=0; i < b; i ++)
                    if (range >= (min_dist + i * distBucket) && range <= (min_dist + (i+1)*distBucket))
                        bucket = i + 1;
            }
            else
            {
                bucket = -1;
            }
            if (bucket > 0)
            {
                iCount = imageCount.at<int>(rowIdn, bucket);
                imageCount.at<int>(rowIdn, bucket) = iCount + 1;
                // <<rowIdn<<" "<<bucket<<" "<< imageCount.at<int>(rowIdn, bucket)<<std::endl;
            }
            if (range < sensorMinimumRange)
                continue;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;
        }
        
        // Normalize each ring
        int sumRing;
        for (size_t i = 0; i < N_SCAN; ++i){
            sumRing = 0;

            for (size_t j = 0; j < b; ++j)
            {
                sumRing += imageCount.at<int>(i, j);
            }                
            for (size_t j = 0; j < b; ++j)
            {
                imageOne.at<float>(i, j) = (float) imageCount.at<int>(i, j) / (float) sumRing * 255.0;
            }                
        }

        cv::flip(imageOne, imageOne, 0);
        auto tensor_image = torch::from_blob(imageOne.data, { imageOne.rows, imageOne.cols}, at::kByte);
        tensor_image.unsqueeze_(0);
        tensor_image = tensor_image.toType(c10::kFloat);
        tensor_image.to(c10::DeviceType::CUDA);
       
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        const at::Tensor output = descriptor.forward(inputs).toTensor();
        
        return output;
    }

    void makeAndSaveLocNet(const pcl::PointCloud<PointType>::Ptr laserCloudIn, int nodeId)
    { 
        auto output = makeDescriptor(laserCloudIn);
        featureFile << nodeId << " ";
        for (int i = 0; i < featureSize; i++)
            featureFile << output[0][i].item<float>() << " "; 
        
        featureFile <<"\n";
    
    }   
    
    //// 
    void loadFeatureCloud()
    {
        ifstream openFile(featurePath.data());        
        std::vector<point<float, featureSize>> points;
        if (openFile.is_open())
        {
            string line;
            while(getline(openFile, line))
            {
                std::istringstream iss(line);
                float f;
                std::vector<float> values;
                while (iss >> f)
                {
                    values.push_back(f);
                }                        
                points.push_back( point<float, featureSize> (begin(values), end(values))); 
            }
            openFile.close();
        }

        tree.reset(new kdtree<float, featureSize>(std::begin(points), std::end(points)));
        // point8d n = tree.nearest({0.0, -0.07963, 0.0193815, -0.074069, -0.0861626, -4.3218, 0.0678813, 15.4607, -0.00420333});
        ROS_INFO("Loaded features");
    }

    void findCandidates(at::Tensor feature, int& nearest, float& distance)
    {
        std::cout<< "intput tensor: " <<feature<<std::endl;
        
        std::vector<float> values;
        values.push_back(0.0);
        for (int i=0; i < featureSize; i ++)
        {
            values.push_back(feature[0][i].item<float>());
        }                        
        
        point<float, featureSize> n = tree->nearest(point<float, featureSize> (begin(values), end(values)));
        nearest = n.get_index();
        distance = tree->distance();
    }
};
}
#endif