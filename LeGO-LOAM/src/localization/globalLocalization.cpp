#include "descriptor.h"
#include <fstream>
#include <ostream>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/MultiArrayDimension.h>


#include <torch/torch.h>
#include <torch/script.h> 

using namespace lego_loam;

using PointType = pcl::PointXYZI;
namespace lego_loam
{
    
class GlobalLocalization : public nodelet::Nodelet {

private:
    LocNetManager *locnetManager;

    ros::Subscriber subPointCloud;
    ros::Subscriber subPose;
    ros::Publisher pubMapCloud;
    ros::Publisher pubInitialize;
    ros::Timer timer;
    std::mutex mtx;

    string modelPath;
    string featureSavePath;
    string keyPosePath;
    double searchRadius;
    pcl::PointCloud<PointType>::Ptr globalMap;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    std::vector<std::vector<float>> keyPoses;
    
    std::vector<float> currPose;
    bool initialized = false;
    int check_cnt = 0;
    int skip_cnt = 1;

public:
    GlobalLocalization() = default;
    void onInit()
    {
        currPose.reserve(3);
        ROS_INFO("\033[1;32m---->\033[0m Global Localization Started.");  

        ros::NodeHandle nh = getNodeHandle();
		ros::NodeHandle nhp = getPrivateNodeHandle();
        // LOADING MODEL
        nhp.param<std::string>("feature_save_path", featureSavePath, "C:\\Users\\Haeyeon Kim\\Desktop\\lego_loam_result\\feature_lists.txt"); 
        locnetManager = new LocNetManager(featureSavePath, false);
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        nhp.param<std::string>("model_path", modelPath, "/home/haeyeon/model.pt"); 

        nhp.param<std::string>("key_pose_path", keyPosePath, "/home/haeyeon/key_poses.txt"); 
        nhp.param<double>("search_radius", searchRadius, 10.0); 
        nhp.param<int>("skip_cnt", skip_cnt, 1); 
        locnetManager->loadModel(modelPath);
        locnetManager->loadFeatureCloud();
        loadKeyPose();     
        pubInitialize = nh.advertise<std_msgs::Float32MultiArray>("/initialize_data", 1);
        
        subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &GlobalLocalization::handlePointCloud, this);        
        subPose = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1, &GlobalLocalization::updatePose, this);        
    }

    void updatePose(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        currPose[0] = msg->pose.pose.position.x;
        currPose[1] = msg->pose.pose.position.y;
        currPose[2] = msg->pose.pose.position.z;
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
        ROS_INFO("Loaded pose list: %d", keyPoses.size()); 
    }

    void handlePointCloud(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        pcl::PointCloud<PointType>::Ptr laserCloudRaw(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*msg, *laserCloudRaw);

        check_cnt ++;
        if (!initialized)
        {
            // initialized = true;
            std_msgs::Float32MultiArray initialPoseMsg;
            auto locnetFeature = locnetManager->makeDescriptor(laserCloudRaw);
            int nnode;
            float distance;

            locnetManager->findCandidates(locnetFeature, nnode, distance); 
            std::cout<<"Nearest: "<<nnode<<" distance: "<< distance<<std::endl;

            std::cout<<"Pose: "<<keyPoses[nnode]<<std::endl;
            std::vector<float> poseData(3); 

            if (distance < searchRadius)
            {
                poseData[0] = keyPoses[nnode][1]; // x
                poseData[1] = keyPoses[nnode][2]; // y
                poseData[2] = keyPoses[nnode][3]; // z
                
                initialPoseMsg.data = poseData; // data: x pose, y pose, z pose
                initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                initialPoseMsg.layout.dim[0].label = "distance";
                initialPoseMsg.layout.dim[0].size = distance;
                
                pubInitialize.publish(initialPoseMsg); 
                ROS_INFO_ONCE("Initialized");
                initialized = true;
            }
        }
        // After initialized: check distance between the previous cloud
        
        else if (check_cnt % skip_cnt == 0)
        {   
            check_cnt = 0;
            std_msgs::Float32MultiArray initialPoseMsg;
            auto locnetFeature = locnetManager->makeDescriptor(laserCloudRaw);
            int nnode;
            float distance;
            locnetManager->findCandidates(locnetFeature, nnode, distance); 
            std::vector<float> poseData(3); // maximum 10 candidates

            if (distance < searchRadius)
            {
                poseData[0] = keyPoses[nnode][1]; // x
                poseData[1] = keyPoses[nnode][2]; // y
                poseData[2] = keyPoses[nnode][3]; // z
            
                // calculate the distance between the previous pose
                float dist = 0;
                for (int i = 0; i < 3; i ++)
                    dist += pow(currPose[i]-poseData[i], 2);

                std::cout<<"distance: "<<sqrt(dist)<<std::endl;
                if (!isinf(dist))
                {
                //     std::cout<<"Might be kidnapped!"<<sqrt(dist)<<std::endl;
                    initialPoseMsg.data = poseData; // data: x pose, y pose, z pose
                    initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                    initialPoseMsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
                    initialPoseMsg.layout.dim[0].label = "kd_distance";
                    initialPoseMsg.layout.dim[0].size = distance;
                    initialPoseMsg.layout.dim[1].label = "prev_distance";
                    initialPoseMsg.layout.dim[1].size = dist;
                    
                    pubInitialize.publish(initialPoseMsg); 
                }

            }
        }
        
    }
};
}
PLUGINLIB_EXPORT_CLASS(lego_loam::GlobalLocalization, nodelet::Nodelet)
