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

using namespace gtsam;
using PointType = pcl::PointXYZI;

class globalLocalization{

private:
    NonlinearFactorGraph gtSAMgraph;
    LocNetManager *locnetManager;

    ros::NodeHandle nh;
    ros::Subscriber subLaserCloudRaw;
    ros::Publisher pubMapCloud;

    string modelPath;
    string featureCloudPath;
    double featureDistBoundary;
    pcl::PointCloud<PointType>::Ptr globalMap;
    pcl::PointCloud<PointType>::Ptr featureCloud;

    bool loaded_map = false;            

public:
    globalLocalization():nh("~")
    {

        vector<BetweenFactor<Pose3>> betweenFactorList;
        ifstream ifs;
        ifs.open("/home/haeyeon/Cocel/between_factor.ros",  ios::binary);   
        ifs.read((char *)&betweenFactorList, sizeof(betweenFactorList));   
        // ROS_INFO("Loaded between factor sized %d", sizeof(betweenFactorList)); 

        // vector<PriorFactor<Pose3>> priorFactorList;
        // ifstream ifs2("/home/haeyeon/Cocel/prior_factor.ros", ios::binary);   
        // ifs2.read((char *)&priorFactorList, sizeof(priorFactorList));   
        // ROS_INFO("Loaded prior factor sized %d", sizeof(priorFactorList)); 

        // nh.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 
        // nh.param<std::string>("feature_cloud_path", featureCloudPath, "/home/haeyeon/locnet_features.pcd"); 
        // nh.param<double>("feature_dist_boundary", featureDistBoundary, 1.0); 
        
        // // LOADING MODEL
        // locnetManager = new LocNetManager();    
        // globalMap.reset(new pcl::PointCloud<PointType>());
        // locnetManager->loadModel(modelPath);
        // locnetManager->loadFeatureCloud(featureCloudPath);

        // loadMapAndGraph();
        // subLaserCloudRaw = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, &globalLocalization::laserCloudRawHandler, this);        
        // pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
    }


    void loadMapAndGraph()
    {
        // load map
        if (pcl::io::loadPCDFile<PointType> ("/home/haeyeon/Cocel/lego_loam_map.pcd", *globalMap) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        loaded_map = true;
        std::cout<< "Loaded Map Cloud"<<std::endl; 

        // load graph
        // ifstream ifs("/home/haeyeon/Cocel/result_gtsam_graph.ros", ios::binary);   
        // ifs.read((char *)&gtSAMgraph, sizeof(gtSAMgraph));   
        // ROS_INFO("Loaded GTSAM Graph sized %d", sizeof(gtSAMgraph)); 

        vector<BetweenFactor<Pose3>> betweenFactorList;
        ifstream ifs1("/home/haeyeon/Cocel/between_factor.ros", ios::binary);   
        ifs1.read((char *)&betweenFactorList, sizeof(betweenFactorList));   
        ROS_INFO("Loaded between factor sized %d", sizeof(betweenFactorList)); 

        vector<PriorFactor<Pose3>> priorFactorList;
        ifstream ifs2("/home/haeyeon/Cocel/prior_factor.ros", ios::binary);   
        ifs2.read((char *)&priorFactorList, sizeof(priorFactorList));   
        ROS_INFO("Loaded prior factor sized %d", sizeof(priorFactorList)); 


    }

    void publishMap()
    {
        if (loaded_map)
        {
            sensor_msgs::PointCloud2 mapCloudMsg;
            pcl::toROSMsg(*globalMap, mapCloudMsg);
            mapCloudMsg.header.stamp = ros::Time::now();
            mapCloudMsg.header.frame_id = "/camera_init";
            pubMapCloud.publish(mapCloudMsg);
        }        
    }

    void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        ROS_INFO_ONCE("called raw handler");
        pcl::PointCloud<PointType>::Ptr laserCloudRaw(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*msg, *laserCloudRaw);
        auto descriptor = locnetManager->makeDescriptor(laserCloudRaw);
        
        std::vector<int> knnIdx;
        std::vector<float> knnDist;
        std::vector<int> nodeIdList;

        locnetManager->findCandidates(descriptor, featureDistBoundary, knnIdx, knnDist, nodeIdList);
        // get node from gtsam graph
        for (std::size_t i = 0; i < knnIdx.size(); i ++)
        {
            std::cout<<"node at" << nodeIdList[i]<<" "<< std::endl;
            
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
        GL.publishMap();
    }

    return 0;
}
