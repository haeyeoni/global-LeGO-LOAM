cmake_minimum_required(VERSION 2.8.3)
project(lego_loam)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

set(GTSAM_LIBRARY_DIRS "C:/Program\ Files\ (x86)/GTSAM/lib")
set(GTSAM_INCLUDE_DIR "C:/Program\ Files\ (x86)/GTSAM/include")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_sensor_msgs

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cloud_msgs

  nodelet
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

# Find Packages
set(PCL_ROOT "C:/Program\ Files/PCL\ 1.8.1")
set(PCL_INCLUDE_DIRS "C:/Program\ Files/PCL\ 1.8.1/include")
set(PCL_LIBRARIES "C:/Program\ Files/PCL\ 1.8.1/lib")
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cloud_msgs nodelet 
  DEPENDS PCL
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR} 
  ${Boost_INCLUDE_DIRS}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
  ${EIGEN3_LIBRARY_DIR} 
  ${Boost_LIBRARY_DIRS}
)

add_library(imageProjection src/mapping/imageProjection.cpp)
add_dependencies(imageProjection ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(featureAssociation src/mapping/featureAssociation.cpp)
add_dependencies(featureAssociation ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(mapOptimization src/mapping/mapOptimization.cpp)
target_link_libraries(mapOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} GTSAM )

add_library(transformFusion src/mapping/transformFusion.cpp)
target_link_libraries(transformFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(globalLocalization src/localization/globalLocalization.cpp)
target_link_libraries(globalLocalization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} GTSAM)

add_executable(particleFilter src/localization/particleFilter.cpp)
target_link_libraries(particleFilter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${Eigen3_LIBRARIES})


set(CMAKE_PREFIX_PATH "C:/Users/Haeyeon\ Kim/projects/libtorch")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.4")
find_package(Torch REQUIRED)

target_link_libraries(mapOptimization "${TORCH_LIBRARIES}")
target_link_libraries(globalLocalization "${TORCH_LIBRARIES}")