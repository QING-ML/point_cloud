# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    if not correlation:
        #print(data.to_numpy().shape)
        #eigenvalues, eigenvectors = np.linalg.eig(np.cov(data.to_numpy().T))
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(data.T))

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file("/home/zqq/C++ Training/point_cloud/hw1/pythonproject/ModelNet40_2/ply_data_points/chair/train/chair_0001.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    points_mean = np.mean(points, axis=0)
    print(points.shape)
    print(points_mean.shape)
    p_start = np.array([0,0,0]) + points_mean
    p_end = point_cloud_vector + points_mean
    # TODO: 此处只显示了点云，还没有显示PCA
    points_pca = np.append([
        p_start],
        [p_end],
        axis=0
    )
    print(points_pca.shape)
    lines_pca = [
        [0,1],
    ]
    colors = [[1,0,0]]
    line_set_pca = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points_pca),
        lines = o3d.utility.Vector2iVector(lines_pca),
    )
    line_set_pca.colors = o3d.utility.Vector3dVector(colors)
    line_set_pca.scale(100)
    #o3d.visualization.draw_geometries([point_cloud_o3d, line_set_pca])

    #o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # 循环计算每个点的法向量
    #p51 in ppt
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    #find nearest points
    for point_i in point_cloud_o3d.points:

        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_i, 20)
        w, v = PCA(np.asarray(point_cloud_o3d.points)[idx[0:], :])
        normal = v[:,0].T # normal
        normals.append(normal)
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    print(normals.shape)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set_pca])
    #在界面中按n查看normal,+号放大normal


if __name__ == '__main__':
    main()
