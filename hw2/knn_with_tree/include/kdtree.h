#ifndef KDTREE_H
#define KDTREE_H
#include "node.h"
#include "KnnResultSet.h"
#include<pcl/io/pcd_io.h>
#include<algorithm>
#include<math.h>
#include<chrono>

class Kdtree{
public:

    Kdtree(int leaf_size){
        this->leaf_size = leaf_size;

    }

    inline void sort_key_by_value(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<int> &point_indices, int axis){

            std::sort(point_indices.begin(), point_indices.end(),[cloud,axis](int a, int b){
                return cloud->points[a].data[axis] < cloud->points[b].data[axis];
            });


    }



    void kdtree_recursive_build(Node* &root, pcl::PointCloud<pcl::PointXYZI>::Ptr const &cloud, std::vector<int> &point_indices,
                                int axis, int leaf_size);

    void kdtree_recursive_build(Node* &root, std::vector<std::vector<double>> &cloud, std::vector<int> &point_indices,
                                int axis, int leaf_size);

    inline int axis_round_robin(int axis){
        if(axis == 2){
            return 0;
        }
        else{
            return axis + 1;
        }
    }
    void kdtree_radius_search(Node* root, pcl::PointCloud<pcl::PointXYZI>::Ptr const &cloud, RadiusNNResultSet& ResultSet, std::vector<float>& query);

    //int db_size;//size of point cloud
    //int dim = 3; //dimension
    int leaf_size; // number of points in leaf node
    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    //Node* root = NULL;
};
#endif
