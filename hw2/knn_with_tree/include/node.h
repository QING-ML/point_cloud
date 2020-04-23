#ifndef Node_H
#define Node_H
#include <iostream>
#include <vector>

class Node{
public:
    Node(int axis, float value, std::vector<int> &point_indices){
        this->axis = axis;
        this->value = value;
        this->point_indices = point_indices;
    }
    int axis;
    float value = 0.0; //split value
    Node* left = NULL;
    Node* right = NULL;
    std::vector<int> point_indices;

    bool is_leaf(){
        return left==NULL&&right==NULL;
    }

    void debug_node(){
        std::cout<< "axis is "<<axis<< " ,point_indices";
        if(is_leaf()){
            std::cout<<" ,split value: leaf";
        }
        else{
            std::cout <<" ,split value:"<<value<<std::endl;
        }
    }





};
#endif
