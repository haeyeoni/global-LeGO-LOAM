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
    double searchRadius;
    pcl::PointCloud<PointType>::Ptr globalMap;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;

    bool loaded_map = false;  

    ISAM2 *isam;
    Values isamCurrentEstimate;  
    Values initialEstimate;  

public:
    globalLocalization():nh("~")
    {
        // LOADING MODEL
        locnetManager = new LocNetManager();    
        globalMap.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        nh.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 
        nh.param<std::string>("feature_cloud_path", featureCloudPath, "/home/haeyeon/locnet_features.pcd"); 
        nh.param<double>("search_radius", searchRadius, 10.0); 
        locnetManager->loadModel(modelPath);
        locnetManager->loadFeatureCloud(featureCloudPath);
        
        loadMapAndGraph();
        pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
        subLaserCloudRaw = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, &globalLocalization::laserCloudRawHandler, this);        
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

        // load poses list        
        if (pcl::io::loadPCDFile<PointType> ("/home/haeyeon/Cocel/key_poses.pcd", *cloudKeyPoses3D) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read pcd \n");
            return;
        }
        std::cout<< "Loaded pose list"<<std::endl; 
    }

    void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        pcl::PointCloud<PointType>::Ptr laserCloudRaw(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*msg, *laserCloudRaw);
        std::vector<int> searchIdx, nodeList;
        std::vector<float> seachDist;
        auto output = locnetManager->makeDescriptor(laserCloudRaw);

        PointType locnet_feature;
        locnet_feature.x = output[0][0].item<float>();
        locnet_feature.y = output[0][1].item<float>();
        locnet_feature.z = output[0][2].item<float>();

        locnetManager->findCandidates(locnet_feature, searchRadius, searchIdx, seachDist, nodeList);
        Pose3 nodeEstimate;
        for (size_t i = 0; i < nodeList.size(); i++)
        {
            std::cout<<"nodes: "<<nodeList[i]<<std::endl;
            std::cout<<"points: "<<cloudKeyPoses3D->points[nodeList[i]].x<<" "<<cloudKeyPoses3D->points[nodeList[i]].y<<std::endl;
            // nodeEstimate = isamCurrentEstimate.at<Pose3>(nodeList[i]-1);
            // std::cout<<"results x"<<nodeEstimate.translation().x()<<std::endl;
        }
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
