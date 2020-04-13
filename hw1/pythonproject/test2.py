# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud

def PCA(data, correlation=False, sort=True):
    # 作业
    # 屏蔽开始
    if not correlation:
        meanData = np.mean(data, axis=0)
        zeromeanData = np.subtract(data, meanData)
        covData = np.cov(zeromeanData, rowvar=0)
        eigenvalues, eigenvectors = np.linalg.eig(np.mat(covData))
    else:
        print('false')
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
    point_cloud_pynt = PyntCloud.from_file(
        "/home/zqq/C++ Training/point_cloud/hw1/pythonproject/ModelNet40_2/ply_data_points/airplane/train/airplane_0001.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理

    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0]  # 点云主方向对应的向量
    print('eigenvector is', v)
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    points_center = np.array(point_cloud_pynt.centroid)
    x_norm=np.linalg.norm(points, ord=2, axis=None, keepdims=False)
    eig_coord = np.dot(v, np.diag(w))
    points_coord = np.append(points_center, np.array(eig_coord)+points_center)
    points = points_coord.reshape(4, 3)
    print(points.shape)
    np.asarray(points)
    lines = np.asarray([
        [0, 1],
        [0, 2],
        [0, 3],
    ])
    print(lines.shape)
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set,point_cloud_o3d])
    o3d.geometry.LineSet()

if __name__ == '__main__':
    main()
