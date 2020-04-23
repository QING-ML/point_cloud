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
//        db_size = cloud->width * cloud->height;
        //this->cloud = cloud;
//        std::vector<int> point_indices;
//        for(int i = 0; i < db_size; i++){
//            point_indices.push_back(i);
//        }

//        auto start_construct = std::chrono::system_clock::now();
//        kdtree_recursive_build(this->root, cloud, point_indices, 0, leaf_size);
//        auto end_construct = std::chrono::system_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_construct - start_construct);
//        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_construct - start_construct);
//        std::cout<<"Time for kdtree construction is "<<double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den<<std::endl;

    }

    inline void sort_key_by_value(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<int>& point_indices, int axis){

            std::sort(point_indices.begin(), point_indices.end(),[cloud,axis](int a, int b){
                return cloud->points[a].data[axis] < cloud->points[b].data[axis];
            });


    }



    void kdtree_recursive_build(Node* &root, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<int> point_indices,
                           int axis, int leaf_size){
        std::cout<<"before create new root, the axis is :"<<axis
                <<" ,points size is :"<<point_indices.size()<<std::endl;
        if(root == NULL){
            //Node new_root(axis, 0.0, point_indices);
            root = new Node(axis, 0.0, point_indices);
            std::cout<<"creat new NODE"<<std::endl;
        }
//        Node new_root(axis, 0.0, point_indices);
//        root = &new_root;
        root->debug_node();
        std::cout<<"after creat new root : "<<std::endl;
        //root = new Node(axis, 0.0 ,point_indices);
        if(point_indices.size() < leaf_size){

            return;
        }
        sort_key_by_value(cloud, point_indices, axis);



        int middle_left_idx = ceil(point_indices.size()/2) - 1;
        int middle_left_point_idx = point_indices[middle_left_idx];
        int middle_left_point_value = cloud->points[middle_left_point_idx].data[axis];

        int middle_right_idx = middle_left_idx + 1;
        int middle_right_point_idx = point_indices[middle_right_idx];
        int middle_right_point_value = cloud->points[middle_right_point_idx].data[axis];
        root->value = (middle_left_point_value + middle_right_point_value) * 0.5;

        //split
        kdtree_recursive_build(root->left, cloud, std::vector<int>(point_indices.begin(), point_indices.begin()+middle_left_idx+1), axis_round_robin(axis),leaf_size);
        kdtree_recursive_build(root->right, cloud, std::vector<int>(point_indices.begin()+middle_right_idx, point_indices.end()), axis_round_robin(axis),leaf_size);

    }

    int axis_round_robin(int axis){
        if(axis == 2){
            return 0;
        }
        else{
            return axis + 1;
        }
    }
    void kdtree_radius_search(Node* root, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, RadiusNNResultSet& ResultSet, std::vector<float>& query){
        if(root == NULL){
            std::cout<<"kdtree over leaf!!"<<std::endl;
            return;
        }

        if(root->is_leaf()){
            for(auto it = root->point_indices.begin(); it < root->point_indices.end(); it++){
                float dist = std::sqrt(std::pow((cloud->points[*it].x - query[0]),2) +
                        std::pow((cloud->points[*it].y - query[1]),2) +
                        std::pow((cloud->points[*it].z - query[2]),2));
                ResultSet.add_point(dist, *it);
            }
            return;
        }

        if(query[root->axis] <= root->value){
            kdtree_radius_search(root->left, cloud, ResultSet, query);
            if(std::fabs(query[root->axis] - root->value) < ResultSet.worst_dist){
                kdtree_radius_search(root->right, cloud, ResultSet, query);
            }
        }
        else{
            kdtree_radius_search(root->right, cloud, ResultSet, query);
            if(std::fabs(query[root->axis] - root->value) < ResultSet.worst_dist){
                kdtree_radius_search(root->left, cloud, ResultSet, query);
            }
        }

    }

    //int db_size;//size of point cloud
    //int dim = 3; //dimension
    int leaf_size; // number of points in leaf node
    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    //Node* root = NULL;
};
#endif
