#include "descriptor.h"
#include <fstream>
#include <ostream>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

using namespace gtsam;
using PointType = pcl::PointXYZI;

class globalLocalization{

private:
    NonlinearFactorGraph gtSAMgraph;
    LocNetManager *locnetManager;

    ros::NodeHandle nh;
    ros::Subscriber subPointCloud;
    ros::Publisher pubMapCloud;
    ros::Publisher pubInitialize;
    string modelPath;
    string featureCloudPath;
    double searchRadius;
    pcl::PointCloud<PointType>::Ptr globalMap;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    ISAM2 *isam;
    Values isamCurrentEstimate;  
    Values initialEstimate;  
    bool initialized = false;

public:
    globalLocalization():nh("~")
    {
        // LOADING MODEL
        locnetManager = new LocNetManager();    
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        nh.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 
        nh.param<std::string>("feature_cloud_path", featureCloudPath, "/home/haeyeon/locnet_features.pcd"); 
        nh.param<double>("search_radius", searchRadius, 10.0); 
        locnetManager->loadModel(modelPath);
        locnetManager->loadFeatureCloud(featureCloudPath);
        
        // load poses list        
        if (pcl::io::loadPCDFile<PointType> ("/home/haeyeon/Cocel/key_poses.pcd", *cloudKeyPoses3D) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        std::cout<< "Loaded pose list"<<std::endl; 

        pubInitialize = nh.advertise<std_msgs::Float32MultiArray>("/initialize_data", 1);
        subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &globalLocalization::handlePointCloud, this);        
    }

    void handlePointCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        if (!initialized)
        {
            // initialized = true;
            std_msgs::Float32MultiArray initialPoseMsg;
            pcl::PointCloud<PointType>::Ptr laserCloudRaw(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*msg, *laserCloudRaw);
            std::vector<int> searchIdx, nodeList;
            std::vector<float> searchDist;
            auto output = locnetManager->makeDescriptor(laserCloudRaw);

            PointType locnetFeature;
            locnetFeature.x = output[0][0].item<float>();
            locnetFeature.y = output[0][1].item<float>();
            locnetFeature.z = output[0][2].item<float>();
            locnetManager->findCandidates(locnetFeature, searchRadius, searchIdx, searchDist, nodeList);
            if (nodeList.size() > 0)
            {
                float similaritySum = 0;

                std::vector<float> poseData(3*10, 0); // maximum 10 candidates
                for (size_t i = 0; i < nodeList.size(); i++)
                {
                    poseData[3*i + 0] = cloudKeyPoses3D->points[nodeList[i]].x;
                    poseData[3*i + 1] = cloudKeyPoses3D->points[nodeList[i]].y;
                    poseData[3*i + 2] = searchDist[i];
                    similaritySum += searchDist[i];
                }

                initialPoseMsg.data = poseData; // data: x pose, y pose, similarity term
                initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                initialPoseMsg.layout.dim[0].label = "similarity_sum";
                initialPoseMsg.layout.dim[0].size = similaritySum;
                initialPoseMsg.layout.dim[1].label = "length";
                initialPoseMsg.layout.dim[1].size = nodeList.size();
                pubInitialize.publish(initialPoseMsg); 
                initialized = true;
            }
        }
    }



};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("Global Localization Started.");

    globalLocalization GL;
    
    while (ros::ok())
    {
        ros::spinOnce();
    }

    return 0;
}
