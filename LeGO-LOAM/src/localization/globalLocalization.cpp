#include "descriptor.h"
#include <fstream>
#include <ostream>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

using namespace lego_loam;

using PointType = pcl::PointXYZI;
namespace lego_loam
{
    
class GlobalLocalization : public nodelet::Nodelet {

private:
    LocNetManager *locnetManager;

    ros::Subscriber subPointCloud;
    ros::Publisher pubMapCloud;
    ros::Publisher pubInitialize;
    ros::Timer timer;

    string modelPath;
    string featureSavePath;
    string keyPosePath;
    double searchRadius;
    pcl::PointCloud<PointType>::Ptr globalMap;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    std::vector<std::vector<float>> keyPoses;
   
    bool initialized = false;

public:
    GlobalLocalization() = default;
    void onInit()
    {
        ROS_INFO("\033[1;32m---->\033[0m Global Localization Started.");  

        ros::NodeHandle nh = getNodeHandle();
		ros::NodeHandle nhp = getPrivateNodeHandle();
        // LOADING MODEL
        nhp.param<std::string>("featureSavePath", featureSavePath, "C:\\Users\\Haeyeon Kim\\Desktop\\lego_loam_result\\feature_lists.txt"); 
        locnetManager = new LocNetManager(featureSavePath, false);
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        nhp.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 

        nhp.param<std::string>("key_pose_path", keyPosePath, "/home/haeyeon/key_poses.txt"); 
        nhp.param<double>("search_radius", searchRadius, 10.0); 
        locnetManager->loadModel(modelPath);
        locnetManager->loadFeatureCloud();
        loadKeyPose();
        
        

        pubInitialize = nh.advertise<std_msgs::Float32MultiArray>("/initialize_data", 1);
        subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &GlobalLocalization::handlePointCloud, this);        
    }

    void loadKeyPose()
    {
        ifstream poseFile(keyPosePath.data());
        if (poseFile.is_open())
        {
            string line;
            while(getline(poseFile, line))
            {
                std::istringstream iss(line);
                float f;
                std::vector<float> values;
                while (iss >> f)
                    values.push_back(f);
                keyPoses.push_back(values);
            }
            poseFile.close();
        }
        ROS_INFO("Loaded pose list"); 
    }

    void handlePointCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        if (!initialized)
        {
            // initialized = true;
            std_msgs::Float32MultiArray initialPoseMsg;
            pcl::PointCloud<PointType>::Ptr laserCloudRaw(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*msg, *laserCloudRaw);
            auto locnetFeature = locnetManager->makeDescriptor(laserCloudRaw);
            
            int nnode;
            float distance;

            locnetManager->findCandidates(locnetFeature, nnode, distance); 
            std::cout<<"Nearest: "<<nnode<<" distance: "<< distance<<std::endl;
            std::vector<float> poseData(3*10, 0); // maximum 10 candidates

            if (distance < searchRadius)
            {
                poseData[0] = keyPoses[nnode][1]; // x
                poseData[1] = keyPoses[nnode][2]; // y
                poseData[2] = keyPoses[nnode][3]; // z

                initialPoseMsg.data = poseData; // data: x pose, y pose, z pose
                initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                initialPoseMsg.layout.dim[0].label = "similarity_sum";
                initialPoseMsg.layout.dim[0].size = distance;
                initialPoseMsg.layout.dim[1].label = "length";
                initialPoseMsg.layout.dim[1].size = 1;
                pubInitialize.publish(initialPoseMsg); 
                ROS_INFO_ONCE("Initialized");
                initialized = true;
            }
        }
    }
};
}
PLUGINLIB_EXPORT_CLASS(lego_loam::GlobalLocalization, nodelet::Nodelet)
