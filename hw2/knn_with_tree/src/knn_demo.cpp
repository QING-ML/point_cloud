#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>

int main(int argc, char** argv){
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if(pcl::io::loadPCDFile<pcl::PointXYZI>("/home/zqq/C++ Training/point_cloud/hw2/velodyne/pcd/0000000000.pcd", *cloud)==-1){
        PCL_ERROR("could not load file");
        return -1;
    }
    std::cout<<"load"<<cloud->width<<","<<cloud->height<<"data points with the following fields:"<<std::endl;
    for(auto i = 0; i < 2; i++){
        std::cout<<""<<cloud->points[i].x
                 <<""<<cloud->points[i].y
                 <<""<<cloud->points[i].z;
    }
    return 0;
}
