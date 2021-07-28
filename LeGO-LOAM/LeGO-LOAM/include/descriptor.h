#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

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


using PointType = pcl::PointXYZI;

class LocNetManager
{

private:
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeFeatures;
    pcl::PointCloud<PointType>::Ptr featureCloud;
    torch::jit::script::Module descriptor;

public: 
    LocNetManager() 
    {
        kdtreeFeatures.reset(new pcl::KdTreeFLANN<PointType>());
        featureCloud.reset(new pcl::PointCloud<PointType>());
    }
    
    /// Mapping
    void loadModel(string modelPath)
    {
        try {
            descriptor = torch::jit::load(modelPath);
        }
        catch (const c10::Error& e) {
            ROS_ERROR("error loading the model\n: %s", e);	    
        }
        ROS_INFO("Success loading model");    
    }

    at::Tensor makeDescriptor(const pcl::PointCloud<PointType>::Ptr laserCloudIn)
    {
        float verticalAngle, horizonAngle, range, prev_range, del_range;
        size_t rowIdn, columnIdn, index, cloudSize; 

        int minDist = 1;
        int maxDist = 81;
        int maxDelDist = 41;

        int b = 80;
        int distBucket, bucket, iCount;
        
        cv::Mat imageCount = cv::Mat(N_SCAN, b, CV_32S, cv::Scalar::all(0));
        cv::Mat imageOne = cv::Mat(N_SCAN, b, CV_32F, cv::Scalar::all(0));
        
        cv::Mat imageDelCount = cv::Mat(N_SCAN, b, CV_32S, cv::Scalar::all(0.0));
        cv::Mat imageDelOne = cv::Mat(N_SCAN, b, CV_32F, cv::Scalar::all(0.0));
        

        cloudSize = laserCloudIn->points.size();
        PointType thisPoint;  
        thisPoint.x = laserCloudIn->points[-1].x;
        thisPoint.y = laserCloudIn->points[-1].y;
        thisPoint.z = laserCloudIn->points[-1].z;
        prev_range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y); // + thisPoint.z * thisPoint.z);
              
        for (size_t i = 0; i < cloudSize; ++i) {

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // find the row and column index in the iamge for this point

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

            // Range
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y); // + thisPoint.z * thisPoint.z);
            distBucket = (maxDist - minDist) / b;
            if (range > minDist && range < maxDist)
            {
                for (int i=0; i < b; i ++)
                    if (range >= (minDist + i * distBucket) && range <= (minDist + (i+1)*distBucket))
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
            }

            // Del image 
            del_range = fabs(prev_range - range);
            distBucket = (maxDelDist - minDist) / b;
            if (del_range > minDist && del_range < maxDelDist)
            {
                for (int i=0; i < b; i ++)
                    if (del_range >= (minDist + i * distBucket) && del_range <= (minDist + (i+1)*distBucket))
                        bucket = i + 1;
            }
            else
            {
                bucket = -1;
            }
    
            if (bucket > 0)
            {
                iCount = imageDelCount.at<int>(rowIdn, bucket);
                imageDelCount.at<int>(rowIdn, bucket) = iCount + 1;
            }

            prev_range = range;
        }
        
        // Normalize each ring
        int sumRing, sumDelRing;
        for (size_t i = 0; i < N_SCAN; ++i){
            sumRing = 0;
            sumDelRing = 0;

            for (size_t j = 0; j < b; ++j)
            {
                sumRing += imageCount.at<int>(i, j);
                sumDelRing += imageDelCount.at<int>(i,j);
            }    
            
            for (size_t j = 0; j < b; ++j)
            {
                imageOne.at<float>(i, j) = (float) imageCount.at<int>(i, j) / (float) sumRing;
                imageDelOne.at<float>(i, j) = (float) imageDelCount.at<int>(i, j) / (float) sumDelRing;   
            }                
        }
        cv::flip(imageOne, imageOne, 0);
        cv::flip(imageDelOne, imageDelOne, 0);     

        cv::Mat merge = cv::Mat(imageOne.size(), CV_32FC2);
        std::vector<cv::Mat>channels;
        channels.push_back(imageOne);
        channels.push_back(imageDelOne);
        cv::merge(channels, merge);   
       
        auto tensor_image = torch::from_blob(merge.data, { merge.rows, merge.cols, merge.channels() }, at::kByte);
        tensor_image = tensor_image.permute({ 2,0,1 });
        tensor_image.unsqueeze_(0);
        tensor_image = tensor_image.toType(c10::kFloat);
        tensor_image.to(c10::DeviceType::CUDA);
       

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        at::Tensor output = descriptor.forward(inputs).toTensor();
        
        return output;
    }

    void makeAndSaveLocNet(const pcl::PointCloud<PointType>::Ptr laserCloudIn, int nodeId)
    { 
        auto output = makeDescriptor(laserCloudIn);

        PointType locnet_feature;
        locnet_feature.x = output[0][0].item<float>();
        locnet_feature.y = output[0][1].item<float>();
        locnet_feature.z = output[0][2].item<float>();
        locnet_feature.intensity = nodeId;

        featureCloud->push_back(locnet_feature);    
        pcl::io::savePCDFileASCII("/home/haeyeon/Cocel/feature_cloud.pcd", *featureCloud);
    }   
    
    ~LocNetManager() 
    {
        // Save Feature Cloud as .pcd file
        
    };
    
    //// Localization
    void loadFeatureCloud(string featurePath)
    {
        if (pcl::io::loadPCDFile<PointType> (featurePath, *featureCloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
        }
        std::cout<< "Loaded Feature Cloud"<<std::endl;        
    }

    void findCandidates(PointType inputFeature)
    {

    }

};