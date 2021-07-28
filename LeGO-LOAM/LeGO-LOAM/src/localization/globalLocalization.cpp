
#include "descriptor.h"

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

class globalLocalization{
private:
    LocNetManager *locnetManager;
    ros::NodeHandle nh;
    string model_path;
    string feature_cloud_path;

public:
    globalLocalization():nh("~")
    {
        // LOADING MODEL
        locnetManager = new LocNetManager();
        nh.param<std::string>("model_path", model_path, "/home/haeyeon/model.pt"); 
        nh.param<std::string>("feature_cloud_path", feature_cloud_path, "/home/haeyeon/locnet_features.pcd"); 

        locnetManager->loadModel(model_path);
        locnetManager->loadFeatureCloud(feature_cloud_path);

    }


};
