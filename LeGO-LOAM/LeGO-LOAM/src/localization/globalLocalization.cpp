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
    ros::Publisher pubMapCloud;

    string modelPath;
    string featureCloudPath;
    pcl::PointCloud<PointType>::Ptr globalMap;

    bool loaded_map = false;            

public:
    globalLocalization():nh("~")
    {
        // LOADING MODEL
        locnetManager = new LocNetManager();    
        globalMap.reset(new pcl::PointCloud<PointType>());
        nh.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 
        nh.param<std::string>("feature_cloud_path", featureCloudPath, "/home/haeyeon/locnet_features.pcd"); 
        locnetManager->loadModel(modelPath);
        locnetManager->loadFeatureCloud(featureCloudPath);
        
        loadMapAndGraph();
        pubMapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 2);
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
        ifstream ifs("/home/haeyeon/Cocel/result_gtsam_graph.ros", ios::binary);   
        ifs.read((char *)&gtSAMgraph, sizeof(gtSAMgraph));   
        std::cout<< "Loaded GTSAM Graph"<<std::endl; 
        // gtSAMgraph.print();
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
