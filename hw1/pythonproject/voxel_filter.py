# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud

def hashvalue(D, grid_index):
    # D: Dimension of grid
    # grid_index: list of hx, hy, hz
    return grid_index[0] + grid_index[1]*D[0] + grid_index[2]*D[0]*D[1]

def selectpoint(id_list, point_cloud, centroid = True ):
    # bool centroid : true: with centroid selection, false: with random selection
    point_cloud = np.asarray(point_cloud)
    #print("point_cloud", point_cloud.shape)
    points=[]
    for id in id_list:
        id = int(id)
        #print("id",id)
        points.append(np.asarray(point_cloud)[id, :])
    points = np.asarray(points)
    if centroid:
        point = np.mean(np.asarray(points), axis=0)
    else:
        point = points[np.random.randint(low=0, high=points.shape[0], size=8)[2], :]
    print("shape of centorid point",point.shape)
    return point


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云 pandas frame
#     leaf_size: voxel尺寸

def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    #ppt65, ppt69
    max_list = np.amax(point_cloud, axis=0)
    print(max_list.shape)
    min_list = np.amin(point_cloud, axis=0)

    #Dimension of grid D
    D = (max_list - min_list)/leaf_size
    print("D shape is ")
    print(D.shape)
    #voxel index

    index_hash_list = []
    for index, row in point_cloud.iterrows():
        index_hash = []
        index_hash.append(index)
        index_hash.append(hashvalue(D,np.floor((row.to_numpy() - min_list)/leaf_size)))
        index_hash_list.append(index_hash)

    print("index_hash_list shape or element A:",np.asarray(index_hash_list)[:,0])
    index_hash_list = np.asarray(index_hash_list)
    print(index_hash_list[:,1].argsort())
    #sorting
    index_hash_list = index_hash_list[index_hash_list[:,1].argsort(),:]
    print("index_hash_list shape or element B:", np.asarray(index_hash_list)[:, 1])
    #downsampling

    threshold = index_hash_list[0, 1]
    id_list = []


    for pair_index_hash in index_hash_list:
        #print(pair_index_hash.shape)
        if pair_index_hash[1] <= threshold:
            id_list.append(pair_index_hash[0])
        else:
            filtered_points.append(selectpoint(id_list, point_cloud, centroid= False))
            threshold = pair_index_hash[1]
            id_list = []
            id_list.append(pair_index_hash[0])
    filtered_points.append(selectpoint(id_list, point_cloud, centroid= False))

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print(filtered_points.shape)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/zqq/C++ Training/point_cloud/hw1/pythonproject/ModelNet40_2/ply_data_points/chair/train/chair_0001.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    #points(2382,3)
    print(point_cloud_pynt.points.shape)
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 2.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
