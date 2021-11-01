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
#include <pcl/kdtree/kdtree_flann.h>


#include <torch/torch.h>
#include <torch/script.h> 
#include "utility.h"
#include "feature_kdtree.h"


using PointType = pcl::PointXYZI;

namespace lego_loam
{

class LocNetManager
{

private:
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeFeatures;
    pcl::PointCloud<PointType>::Ptr featureCloud;

    torch::jit::script::Module descriptor;
    std::ofstream featureFile;
    std::string featurePath;
    // kdtree featureKDTree;

public: 
    LocNetManager(std::string path, bool mapping) 
    {
        kdtreeFeatures.reset(new pcl::KdTreeFLANN<PointType>());
        // featureCloud.reset(new pcl::PointCloud<PointType>());
        featurePath = path;
        if (mapping)
            featureFile.open(path);        
    }
    ~LocNetManager()
    {
        featureFile.close();
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
        featureFile.open(featurePath);
        auto output = makeDescriptor(laserCloudIn);
        featureFile << output[0][0].item<float>() << " " 
                    << output[0][1].item<float>() << " " 
                    << output[0][2].item<float>() << " " 
                    << output[0][3].item<float>() << " " 
                    << output[0][4].item<float>() << " " 
                    << output[0][5].item<float>() << " " 
                    << output[0][6].item<float>() << " " 
                    << output[0][7].item<float>() << " " 
                    << nodeId
                    << "\n";
        featureFile.close();
        // pcl::io::savePCDFileASCII(feature_cloud_path, *featureCloud);
    }   
    
    //// Localization
    void loadFeatureCloud()
    {
        ifstream openFile(featurePath.data());
        std::vector<point<float, 9>> points;
        
        if (openFile.is_open())
        {
            string line;
            std::vector<float> values;
            while(getline(openFile, line))
            {
                std::istringstream iss(line);
                float f;
                while (iss >> f)
                    values.push_back(f);

                point<float, 9> ft({values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]});
                points.push_back(ft);
            }
            openFile.close();
        }
        
        kdtree<float, 8> tree(std::begin(points), std::end(points));
        
        // if (pcl::io::loadPCDFile<PointType> (featurePath, *featureCloud) == -1) //* load the file
        // {
        //     PCL_ERROR ("Couldn't read pcd \n");
        // }
        // std::cout<< "Loaded Feature Cloud"<<std::endl;      
        // kdtreeFeatures->setInputCloud(featureCloud);  
    }

    void findCandidates(PointType inputFeature, double radius, std::vector<int> &searchIdx, std::vector<float> &searchDist, std::vector<int> &nodeList)
    {   
        kdtreeFeatures->radiusSearch(inputFeature, radius, searchIdx, searchDist);
        ROS_INFO("Found %d number of candidates", searchIdx.size());
        for (size_t i = 0; i < searchIdx.size(); i ++)
        {
            nodeList.push_back(featureCloud->points[searchIdx[i]].intensity);

            std::cout<<"key pose: "<<featureCloud->points[searchIdx[i]].intensity<<std::endl;
        }

    }

};
}
#endif