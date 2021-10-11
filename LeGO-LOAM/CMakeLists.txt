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

set(CMAKE_PREFIX_PATH "C:/Users/Haeyeon\ Kim/projects/libtorch")
find_package(Torch REQUIRED)

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


add_library(ImageProjection src/mapping/imageProjection.cpp)
add_dependencies(ImageProjection ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(ImageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(FeatureAssociation src/mapping/featureAssociation.cpp)
add_dependencies(FeatureAssociation ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(FeatureAssociation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(MapOptimization src/mapping/mapOptimization.cpp)
target_link_libraries(MapOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} GTSAM "${TORCH_LIBRARIES}")

add_library(TransformFusion src/mapping/transformFusion.cpp)
target_link_libraries(TransformFusion ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(GlobalLocalization src/localization/globalLocalization.cpp)
target_link_libraries(GlobalLocalization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(ParticleFilter3D src/localization/particleFilter.cpp)
target_link_libraries(ParticleFilter3D ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${Eigen3_LIBRARIES}"${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MapOptimization
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:MapOptimization>)
endif (MSVC)