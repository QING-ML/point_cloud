#ifndef KNNRESULTSET_H
#define KNNRESULTSET_H
#include<vector>

class DistIndex{
public:
    DistIndex(float distance, int index){
        this->distance = distance;
        this->index = index;
    }
    float distance;
    int index;
};

class RadiusNNResultSet{
public:

    RadiusNNResultSet(float radius){
        this->radius = radius;
        this->worst_dist = radius;
    }
    void add_point(float dist, int index){
        comparison_counter += 1;
        if(dist > radius){
            return;
        }

        count += 1;
        dist_index_list.push_back(DistIndex(dist,index));
    }

    float radius;
    int count = 0;
    float worst_dist;
    std::vector<DistIndex> dist_index_list;

    int comparison_counter = 0;


};
#endif
