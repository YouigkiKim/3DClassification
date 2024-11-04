import open3d as o3d
import numpy as np


# train_index_path = /home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/ImageSets_Team_3/train_origin2.txt
# test_index_path  = /home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/ImageSets_Team_3/test.txt

z_offset = []
# index_range = 
for i in range(9):
    points_file = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100000.npy"

    points = np.load(points_file)  # .npy 파일로부터 포인트 로드
    y_min, y_max = -1,1
    roi_points = points[(points[:, 2] >= y_min) & (points[:, 2] <= y_max)]

    # # 1. 포인트 클라우드 로드
    # points = np.load(points_file)  # .npy 파일로부터 포인트 로드
    # y_min, y_max = -5, 5
    # roi_points = points[(points[:, 1] >= y_min) & (points[:, 1] <= y_max)]

    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z 좌표만 가져옴
    pcd.points = o3d.utility.Vector3dVector(roi_points[:, :3])  # x, y, z 좌표만 가져옴

    # RANSAC을 사용하여 바닥면 탐지
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=2000)
    [a, b, c, d] = plane_model
    z_offset.append(d)
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    print(f"z offset : {d}")
    # 바닥면과 비 바닥면 포인트 분리
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # 시각화
    inlier_cloud.paint_uniform_color([0.0, 0, 0])  # 바닥면을 빨간색으로 표시
    outlier_cloud.paint_uniform_color([0, 0.0, 0])  # 나머지 포인트를 초록색으로 표시
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

print(f"z offsfet mean : {np.mean(z_offset)}")

# 바닥면과 비 바닥면 포인트 분리
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# 시각화
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 바닥면을 빨간색으로 표시
outlier_cloud.paint_uniform_color([0, 1.0, 0])  # 나머지 포인트를 초록색으로 표시
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])