cmake_minimum_required(VERSION 2.6)
project(knn_with_tree)
add_compile_options(-std=c++11)

find_package(PCL 1.2 REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME} src/kdtree.cpp src/octree.cpp)

add_executable(knn src/knn_demo.cpp)
target_link_libraries(knn ${PROJECT_NAME} ${PCL_LIBRARIES})

#add_executable(binToPcd src/binToPcd.cpp)
#target_link_libraries(binToPcd ${PROJECT_NAME} ${PCL_LIBRARIES})
