# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture, neighbors
from itertools import cycle, islice
import matplotlib.pyplot as plt
import open3d as o3d
import sys
sys.setrecursionlimit(100000) #提高递归层数上限
from mpl_toolkits.mplot3d import Axes3D


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
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
    return np.asarray(pc_list, dtype=np.float32)

def Ransac(z_filter_arr_idx,data):
    #功能:fitting 一个平面
    #输入: 一个数据data
    #输出: idx-ground points
    s = 3
    e = 0.25
    p = 0.99
    N = int(np.ceil(np.log(1-p)/np.log(1-np.power((1-e),s))))
    print('N:', N)
    tau = 0.5

    model = np.zeros((3,3))
    n_inline = 0
    r_normal = np.zeros(3)

    for n in range(0, N):
        np.random.shuffle(z_filter_arr_idx)
        select_pts = data[z_filter_arr_idx[:3]]
        print('select_pts shape:',select_pts.shape)
        p1p2 = select_pts[1] - select_pts[0]
        p1p3 = select_pts[2] - select_pts[0]
        normal = np.cross(p1p2, p1p3)

        temp_n_inline = 0;
        for pt_idx in z_filter_arr_idx[3:]:
            if np.dot(normal.T,(data[pt_idx] - select_pts[0]))/np.linalg.norm(normal) < tau:
                temp_n_inline = temp_n_inline + 1
        if temp_n_inline > n_inline:
            print("temp n inline points :", temp_n_inline)
            n_inline = temp_n_inline
            model[0] = select_pts[0]
            model[1] = select_pts[1]
            model[2] = select_pts[2]
            r_normal = normal
    result_idx = []
    print("data after ransac:", data.shape)
    for pt_idx in z_filter_arr_idx:
        if np.dot(r_normal.T, (data[pt_idx] - model[0])) / np.linalg.norm(r_normal) < tau:
            #print("distance: ", np.dot(r_normal, (pt_d - model[0])) / np.linalg.norm(r_normal))
            result_idx.append(pt_idx)
    print("ransac result shape: ", np.asarray(result_idx).shape)
    return np.asarray(result_idx)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    print(data.shape)
    data_idx = np.arange(data.shape[0])
    print("data index :", data_idx.shape)
    z_filter_arr_idx =  data_idx[data[:,2] < -1.55]
    print('z filtered data: ',z_filter_arr_idx.shape)
    print("data_2 shape :", data.shape)
    ground_idx = Ransac(z_filter_arr_idx,data)
    segmengted_cloud = np.delete(data, ground_idx, axis=0)


    #non-ground index
    non_ground_indices = np.arange(data.shape[0])
    non_ground_indices = np.delete(non_ground_indices, ground_idx)





    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    print('non-ground points indicies ', non_ground_indices.shape)
    return segmengted_cloud, non_ground_indices

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）


def clustering(data):
    # 作业2
    # 屏蔽开始
    #DBSCAN
    #noise 编号为0
    n_class = 0
    r = 0.1
    min_samples = 35
    idx = np.arange(data.shape[0])
    #nbrs = neighbors.NearestNeighbors(radius=r, algorithm='kd_tree').fit(data[idx])
    result = np.zeros(data.shape[0])

    def DFS(point_idx, neighbor_indices, data, n_class, min_samples, result, r):
        nonlocal idx
        if (neighbor_indices.shape[0] < min_samples):
            dele_idx = np.argwhere(idx == point_idx)
            dele_idx = np.reshape(dele_idx, 1)
            idx = np.delete(idx, dele_idx)
            result[point_idx] = n_class
            return

        result[point_idx] = n_class
        dele_idx = np.argwhere(idx == point_idx)
        #print("point _idx:", point_idx)
        dele_idx = np.reshape(dele_idx, 1)
        #print("dele idx after reshape:", dele_idx)
        idx = np.delete(idx, dele_idx)
        #print("idx shape after delete:", idx.shape)
        # print("DFS neighbor_Indices", neighbor_indices.shape)
        for neighbor_idx in neighbor_indices:
            if(np.argwhere(idx == neighbor_idx)):
                # print("neighbor_idx shape", neighbor_idx
                nbrs = neighbors.NearestNeighbors(radius=r, algorithm='kd_tree').fit(data[idx])
                distances, indices = nbrs.radius_neighbors(data[[neighbor_idx]])
                new_neigh_indices = indices[0][indices[0][:] != neighbor_idx]
                #print("new_neigh_indices shape: ", new_neigh_indices.shape)
                expand_indices = []
                for expand_idx in new_neigh_indices:
                    if(np.argwhere(idx == expand_idx)):
                        expand_indices.append(expand_idx)
                #print("expand_indices shape: ", np.asarray(expand_indices).shape)
                DFS(neighbor_idx, np.asarray(expand_indices), data, n_class, min_samples, result, r)
        return

    while idx.shape[0]:
        #print("idx number is :",idx.shape)
        start_idx = np.random.choice(idx, 1)
        #print("start_idx", start_idx)
        nbrs = neighbors.NearestNeighbors(radius = r, algorithm= 'kd_tree').fit(data[idx])
        #print("start_ idx shape :", start_idx.shape)
        distances, indices =nbrs.radius_neighbors(data[start_idx])
        #print("distance:",  distances)
        #print("indices:" , indices[0].shape)

        if indices[0].shape[0] - 1 < min_samples:
            print("noise point")
            result[start_idx] = 0;
            dele_idx = np.argwhere(idx == start_idx)
            dele_idx = np.reshape(dele_idx, 1)
            #print("dele idx:", dele_idx)
            idx = np.delete(idx, dele_idx)
            #print("idx shape after delete:", idx.shape)
        else:
            n_class = n_class + 1
            print("n_class: ", n_class)
            neigh_indices = indices[0][indices[0][:] != start_idx]
            print("neigh_indices shape:", neigh_indices.shape)
            DFS(start_idx, neigh_indices, data, n_class, min_samples, result, r)



    clusters_index = result

    # 屏蔽结束

    return clusters_index, n_class

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = '/home/zqq/C++ Training/point_cloud/hw2/velodyne/bin' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)
        ax = plt.figure().add_subplot(111, projection = '3d')
        ax.scatter(origin_points[:, 0], origin_points[:, 1], origin_points[:, 2])
        plt.show()

#self write
def calibrarion_test(data):
    index = 0
    idx_list = []
    for pt in data:
        if pt[2] < -1.4:
            idx_list.append(index)
        index = index + 1
    print("index shape", np.asarray(idx_list).shape)
    return idx_list

def plot_clusters_03d(non_ground_indices,segment_points,cluster_index, pcd, n_class):

    print(n_class)
    Num = int(n_class + 1)
    colors = np.array(list(islice(cycle([[55, 126, 184], [255, 127, 0], [77, 175, 74],
                                             [247, 129, 191], [166, 86, 40], [152, 78, 163],
                                             [153, 153, 153], [228, 26, 28], [222, 222, 0]]),
                                      int(max(cluster_index) + 1))))

    print("color shape: ", colors.shape)
    class_indicies_list = []
    for n in range(0, Num):
        class_indicies = np.argwhere(cluster_index == n)
        print("class_indicies", class_indicies.shape)

        pts_idx_list = []
        for idx in class_indicies:
            pts_idx_list.append(non_ground_indices[idx])
        pts_idx_list = np.reshape(pts_idx_list,(np.asarray(pts_idx_list).shape[0],)).astype(int)

        print("pts_idx_list shape : ",pts_idx_list)
        np.asarray(pcd.colors)[pts_idx_list[:], :] = colors[n]
    o3d.visualization.draw_geometries([pcd])









def main2():
    pcd = o3d.io.read_point_cloud("/home/zqq/C++ Training/point_cloud/hw2/velodyne/pcd/0000000020.pcd")
    pcd.paint_uniform_color([0.0, 0.0, 1.0])
    filename = "/home/zqq/C++ Training/point_cloud/hw2/velodyne/bin/0000000020.bin"
    origin_points = read_velodyne_bin(filename)

    #brute ground visulization
    #idx = calibrarion_test(origin_points)
    #print("color shape :", np.asarray(pcd.colors).shape)
    #np.asarray(pcd.colors)[idx[:], :] = [0, 1, 0]
    #o3d.visualization.draw_geometries([pcd])


    #clustering
    segmented_points, non_ground_indices = ground_segmentation(data=origin_points)
    np.savetxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/experiment_data/segmented_points_20.txt', segmented_points)
    np.savetxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/experiment_data/non_ground_indicess_20.txt', non_ground_indices)
    cluster_index, n_class = clustering(segmented_points)
    np.savetxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/experiment_data/cluster_index.txt', cluster_index)
    np.savetxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/experiment_data/n_class.txt', [n_class])
    plot_clusters_03d(non_ground_indices, segmented_points, cluster_index, pcd, n_class)

def plot_cluster_from_load_file():
    pcd = o3d.io.read_point_cloud("/home/zqq/C++ Training/point_cloud/hw2/velodyne/pcd/0000000020.pcd")
    pcd.paint_uniform_color([0.0, 0.0, 1.0])
    segmented_points = np.loadtxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/segmented_points_20.txt')
    print("segmented_points: ", segmented_points.shape)
    non_ground_indices = np.loadtxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/non_ground_indicess_20.txt')
    print('non_ground_indices: ', non_ground_indices.shape)
    cluster_index = np.loadtxt('/home/zqq/C++ Training/point_cloud/hw4/model_fitting/cluster_index.txt')
    print('cluster_index', cluster_index.shape)

    n_class =  np.amax(cluster_index, axis=0)
    print(n_class)
    Num = int(n_class + 1)
    colors = np.array(list(islice(cycle([[55, 126, 184], [255, 127, 0], [77, 175, 74],
                                             [247, 129, 191], [166, 86, 40], [152, 78, 163],
                                             [153, 153, 153], [228, 26, 28], [222, 222, 0]]),
                                      int(max(cluster_index) + 1))))
    colors = np.random.rand(Num, 3)

    print("color shape: ", colors.shape)
    class_indicies_list = []
    for n in range(0, Num):
        class_indicies = np.argwhere(cluster_index == n)
        print("class_indicies", class_indicies.shape)

        pts_idx_list = []
        for idx in class_indicies:
            pts_idx_list.append(non_ground_indices[idx])
        pts_idx_list = np.reshape(pts_idx_list,(np.asarray(pts_idx_list).shape[0],)).astype(int)

        print("pts_idx_list shape : ",pts_idx_list)
        np.asarray(pcd.colors)[pts_idx_list[:], :] = colors[n]
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main2()
    #plot_cluster_from_load_file()
