# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import open3d as o3d
import os
import struct

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32) #不转置为kdtree,转置为octree

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    root_dir = '/home/zqq/C++ Training/point_cloud/hw2/velodyne/bin' # 数据集路径
    cat = os.listdir(root_dir)
    #iteration_num = len(cat)
    iteration_num = 3

    # print("octree --------------")
    # construction_time_sum = 0
    # knn_time_sum = 0
    # radius_time_sum = 0
    # brute_time_sum = 0
    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     db_np = read_velodyne_bin(filename)
    #
    #     begin_t = time.time()
    #     root = octree.octree_construction(db_np, leaf_size, min_extent)
    #     construction_time_sum += time.time() - begin_t
    #
    #     query = db_np[0,:]
    #
    #     begin_t = time.time()
    #     result_set = KNNResultSet(capacity=k)
    #     octree.octree_knn_search(root, db_np, result_set, query)
    #     knn_time_sum += time.time() - begin_t
    #
    #     begin_t = time.time()
    #     result_set = RadiusNNResultSet(radius=radius)
    #     octree.octree_radius_search_fast(root, db_np, result_set, query)
    #     radius_time_sum += time.time() - begin_t
    #
    #     begin_t = time.time()
    #     diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    #     nn_idx = np.argsort(diff)
    #     nn_dist = diff[nn_idx]
    #     brute_time_sum += time.time() - begin_t
    # print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
    #                                                                  knn_time_sum*1000/iteration_num,
    #                                                                  radius_time_sum*1000/iteration_num,
    #                                                                  brute_time_sum*1000/iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        #radius最近邻的数量
        # print("size of result",result_set.size())
        # for Distance_index in result_set.dist_index_list:
        #     print("Index and distance :", Distance_index.index, Distance_index.distance)

        #
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))

    print("o3d test --------------------------------------------------")

    pcd = o3d.io.read_point_cloud("/home/zqq/C++ Training/point_cloud/hw2/velodyne/pcd/0000000003.pcd")
    begin_t = time.time()
    pcd_tree = o3d.geometry.KDTreeFlann(pcd);
    o3d_build_time = time.time() - begin_t

    begin_search_t = time.time()
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[0], 1)
    o3d_search_time =time.time()-begin_t;
    print("creat tree time: build %3f, search %3f" %(o3d_build_time * 1000,
                                         o3d_search_time * 1000) )


if __name__ == '__main__':
    main()