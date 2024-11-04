import argparse
import glob
from pathlib import Path
import pickle
try:
    import open3d as o3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def load_gt_data(label_path, index):
    """GT 데이터 로드 함수 - index.txt 파일 형식 (클래스 이름 포함)"""
    gt_file = Path(f"{label_path}/{index:06d}.txt")

    gt_data = []
    if gt_file.is_file():
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[4:-1])) # 나머지 요소는 좌표와 크기 정보 (float으로 변환)
                xmin, ymin, xmax, ymax = coords[4:7]
                location = coords[11:13]
                rotation = coords[14]
                coords = np.concat( (np.array(location),np.array([xmax-xmin, ymax-ymin, coords[8]]), np.array([rotation])) , axis = 0)
                gt_data.append(coords) # 튜플로 저장 (클래스 이름, 좌표 정보)
        print(gt_data)
        return np.array(gt_data)
    else:
        print("gt file does not exist")
    return None

# 포인트 클라우드 데이터 로드
pcl_path = '/home/ailab/AILabDataset/01_Open_Dataset/03_KITTI/train/lidar'
index = 0
data_array = []

# 여러 파일의 포인트 클라우드를 읽어 리스트에 저장
for i in range(6):
    data = np.fromfile(f"{pcl_path}/{index + i:06d}.bin", dtype=np.float32).reshape(-1, 4)
    data_array.append(data)

# 모든 포인트 클라우드 데이터를 하나의 배열로 병합
data = np.concatenate(data_array, axis=0)  # 행 방향(위-아래)으로 연결

# open3d PointCloud 객체 생성
pcd = o3d.geometry.PointCloud()

# 좌표 (x, y, z) 설정
pcd.points = o3d.utility.Vector3dVector(data[:, :3])

# 강도(intensity) 값이 있는 경우 색상 설정
if data.shape[1] > 3:
    intensities = data[:, 3]  # intensity 값 추출
    colors = np.zeros((data.shape[0], 3))
    colors[:, 0] = intensities / intensities.max()  # intensity를 정규화하여 빨간색 채널에 반영
    pcd.colors = o3d.utility.Vector3dVector(colors)

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd])