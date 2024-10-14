import open3d as o3d
import numpy as np

# 경로 설정
points_file = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100000.npy"
labels_file = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels/00100000.txt"

# 1. 포인트 클라우드 로드
points = np.load(points_file)  # .npy 파일로부터 포인트 로드
print(points.shape)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z 좌표만 가져옴

# 2. 라벨 로드 및 바운딩 박스 그리기 함수
def create_bounding_box(label):
    x, y, z, dx, dy, dz, heading_angle, category_name = label
    if category_name == "Vehicle":
        color = [1, 0, 0]  # 빨간색
    elif category_name == "Pedestrian":
        color = [0, 1, 0]  # 초록색
    elif category_name == "Cyclist":
        color = [0, 0, 1]  # 파란색
    else:
        color = [1, 1, 1]  # 흰색 (기타)

    # 바운딩 박스의 중심 좌표와 회전 적용
    bounding_box = o3d.geometry.OrientedBoundingBox(center=[x, y, z], R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, heading_angle]), extent=[dx, dy, dz])
    bounding_box.color = color
    return bounding_box

# 3. 라벨 파일 파싱
boxes = []
with open(labels_file, 'r') as f:
    for line in f:
        label = line.strip().split()
        label_data = [float(v) for v in label[:-1]] + [label[-1]]  # 마지막은 class 이름
        box = create_bounding_box(label_data)
        boxes.append(box)

# 4. 시각화
o3d.visualization.draw_geometries([point_cloud] + boxes)  # 포인트 클라우드와 바운딩 박스 함께 시각화
