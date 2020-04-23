#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include"kdtree.h"

int main(int argc, char** argv){
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if(pcl::io::loadPCDFile<pcl::PointXYZI>("/home/zqq/C++ Training/point_cloud/hw2/velodyne/pcd/0000000003.pcd", *cloud)==-1){
        PCL_ERROR("could not load file");
        return -1;
    }
    std::cout<<"load"<<cloud->width<<","<<cloud->height<<"data points with the following fields:"<<std::endl;
    for(auto i = 0; i < 1; i++){
        std::cout<<"x: "<<cloud->points[i].x
                 <<" y:  "<<cloud->points[i].y
                 <<" z:"<<cloud->points[i].z
                 <<"//"<<cloud->points[i].data[0]<<std::endl;
        std::cout<<"x: "<<cloud->points[3961].x
                 <<" y:  "<<cloud->points[3961].y
                 <<" z:"<<cloud->points[3961].z
                 <<"//"<<cloud->points[i].data[0]<<std::endl;
    }

    int leaf_size = 32;
    float radius = 1 ;
    Kdtree kdtree(leaf_size);
    std::vector<int> point_indices;
    for(int i = 0; i < cloud->width * cloud->height; i++){
        point_indices.push_back(i);
    }

    Node* root = NULL;


    auto start_construct = std::chrono::system_clock::now();
    kdtree.kdtree_recursive_build(root, cloud, std::move(point_indices), 0, leaf_size);

    auto end_construct = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_construct - start_construct);
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_construct - start_construct);
    std::cout<<"Time for kdtree construction is "<<double(duration.count())<<std::endl;
    if(root != NULL){
        root->debug_node();
        std::cout<<"axis of root: "<< root->axis<<std::endl;
    }
    else{
        std::cout<<"No tree"<<std::endl;
    }


    RadiusNNResultSet ResultSet(radius);
    std::vector<float> query={cloud->points[0].x,
                             cloud->points[0].y,
                             cloud->points[0].z};
    std::cout<< "start search"<<std::endl;
    auto start_search = std::chrono::system_clock::now();
    kdtree.kdtree_radius_search(root, cloud, ResultSet,query);
    auto end_search = std::chrono::system_clock::now();
    auto duration_search = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_construct - start_construct);
    std::cout<<"Time for kdtree search is "<<double(duration_search.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den<<std::endl;
    std::cout<<"number of nearest points index : "<< ResultSet.dist_index_list.size()<<std::endl;

    for(auto Distance_Index:ResultSet.dist_index_list){
        std::cout<<"Index of nearest point: "<<Distance_Index.index
                <<" ,distance to query point: : "<<Distance_Index.distance<<std::endl;
    }

}
