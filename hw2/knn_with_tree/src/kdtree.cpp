#include"kdtree.h"
#include<thread>
void Kdtree::kdtree_recursive_build(Node* &root, pcl::PointCloud<pcl::PointXYZI>::Ptr const &cloud, std::vector<int> &point_indices, int axis, int leaf_size){
//    std::cout<<"before create new root, the axis is :"<<axis
//            <<" ,points size is :"<<point_indices.size()<<std::endl;

    if(root == nullptr){
        root = new Node(axis, 0.0, point_indices);
//        std::cout<<"creat new NODE"<<std::endl;
    }
//        Node new_root(axis, 0.0, point_indices);
//        root = &new_root;
    //root->debug_node();
    //std::cout<<"after creat new root : "<<std::endl;
    //root = new Node(axis, 0.0 ,point_indices);
    if(point_indices.size() < leaf_size){

        return;
    }

    //sort_key_by_value(cloud, point_indices, axis);

    std::sort(point_indices.begin(), point_indices.end(),[&cloud,axis](int a, int b){
        return cloud->points[a].data[axis] < cloud->points[b].data[axis];
    });



    int middle_left_idx = ceil(point_indices.size()/2.0) - 1;
    int middle_left_point_idx = point_indices[middle_left_idx];
    float middle_left_point_value = cloud->points[middle_left_point_idx].data[axis];

    int middle_right_idx = middle_left_idx + 1;
    int middle_right_point_idx = point_indices[middle_right_idx];
    float middle_right_point_value = cloud->points[middle_right_point_idx].data[axis];
    root->value = (middle_left_point_value + middle_right_point_value) * 0.5;

    //split
    std::vector<int> point_indices_left(std::vector<int>(point_indices.begin(), point_indices.begin()+middle_right_idx));
    kdtree_recursive_build(root->left, cloud, point_indices_left, axis_round_robin(axis),leaf_size);
    std::vector<int> point_indices_right(std::vector<int>(point_indices.begin()+middle_right_idx, point_indices.end()));
    kdtree_recursive_build(root->right, cloud, point_indices_right, axis_round_robin(axis),leaf_size);

}

void Kdtree::kdtree_recursive_build(Node *&root,  std::vector<std::vector<double> > &cloud, std::vector<int> &point_indices, int axis, int leaf_size){
    //    std::cout<<"before create new root, the axis is :"<<axis
    //            <<" ,points size is :"<<point_indices.size()<<std::endl;

        if(root == nullptr){

            //std::cout<< "before creat new node size of points_indices: "<< point_indices.size()<<std::endl;
            root = new Node(axis, 0.0, point_indices);
            //std::cout<<"after creat new node size of points_indices: "<< point_indices.size()<<std::endl;


    //        std::cout<<"creat new NODE"<<std::endl;
        }
    //        Node new_root(axis, 0.0, point_indices);
    //        root = &new_root;
        //root->debug_node();
        //std::cout<<"after creat new root : "<<std::endl;
        //root = new Node(axis, 0.0 ,point_indices);
        if(point_indices.size() > leaf_size){



        //sort_key_by_value(cloud, point_indices, axis);


        std::sort(point_indices.begin(), point_indices.end(),[&cloud,&axis](int &a, int &b){
            return cloud[a][axis] < cloud[b][axis];
        });



        size_t middle_left_idx = ceil(point_indices.size()/2.0) - 1;
        int &middle_left_point_idx = point_indices[middle_left_idx];
        double &middle_left_point_value = cloud[middle_left_point_idx][axis];

        size_t middle_right_idx = middle_left_idx + 1;
        int &middle_right_point_idx = point_indices[middle_right_idx];
        double &middle_right_point_value = cloud[middle_right_point_idx][axis];
        root->value = (middle_left_point_value + middle_right_point_value) * 0.5;

        //split
        std::vector<int> point_indices_left(std::vector<int>(point_indices.begin(), point_indices.begin()+middle_right_idx));
        kdtree_recursive_build(root->left, cloud, point_indices_left, axis_round_robin(axis),leaf_size);
        std::vector<int> point_indices_right(std::vector<int>(point_indices.begin()+middle_right_idx, point_indices.end()));
        kdtree_recursive_build(root->right, cloud, point_indices_right, axis_round_robin(axis),leaf_size);


        }

}

void Kdtree::kdtree_radius_search(Node *root, pcl::PointCloud<pcl::PointXYZI>::Ptr const &cloud, RadiusNNResultSet &ResultSet, std::vector<float> &query){
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
