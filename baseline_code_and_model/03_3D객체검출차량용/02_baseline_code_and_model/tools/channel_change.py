import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_bounding_box(x, y, z, dx, dy, dz, heading):
    """Create a bounding box in Open3D."""
    
    rotation = R.from_euler('z', heading).as_matrix()

    box = o3d.geometry.OrientedBoundingBox(
        center=(x, y, z),
        R=rotation,
        extent=(dx, dy, dz)
    )
    return box

# Load point cloud data
points_file = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100000.npy"
points = np.load(points_file)
print(points.shape)

# LiDAR position and FOV angles
lidar_position = np.array([0, 0, 2.1])
fov_min, fov_max = -25, 15  # in degrees

# Convert FOV to radians and divide it into 128 segments
fov_min_rad, fov_max_rad = np.radians(fov_min), np.radians(fov_max)
angle_step = (fov_max_rad - fov_min_rad) / 128

# Extract (x, y, z) coordinates
points_xyz = points[:, :3]

# Compute distance from LiDAR to each point
distances = np.linalg.norm(points_xyz - lidar_position, axis=1)

# Calculate vertical angle for each point
angles = np.arcsin((points_xyz[:, 2] - lidar_position[2]) / distances)

# Initialize a list to store points in selected odd-numbered channels
selected_points = []

# Loop through each odd-numbered segment index
for i in range(1, 128, 2):
    # Define the angle range for the current segment
    lower_angle = fov_min_rad + i * angle_step
    upper_angle = lower_angle + angle_step
    
    # Select points within this angle range
    segment_mask = (angles >= lower_angle) & (angles < upper_angle)

    selected_points.append(points[segment_mask])

# Concatenate all selected points from the odd-numbered segments

filtered_points = np.vstack(selected_points)

# Find min and max z values among filtered points
min_z = np.min(filtered_points[:, 2])
max_z = np.max(filtered_points[:, 2])

print(f"Minimum z value: {min_z}")
print(f"Maximum z value: {max_z}")
print(filtered_points.shape)

# Create Open3D point cloud with filtered points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points[:,:3])

# Set all points to a color (e.g., black)
pcd.colors = o3d.utility.Vector3dVector(np.zeros((filtered_points.shape[0], 3)))

# Load bounding box data from the label file
label_file = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels/00100000.txt"
bboxes = []

with open(label_file, 'r') as f:
    for line in f:
        values = line.strip().split()
        if len(values) == 8:
            x, y, z = map(float, values[:3])  # x, y, z
            dx, dy, dz = map(float, values[3:6])  # dimensions
            heading = float(values[6])  # heading angle
            cls = values[7]  # class

            # Create a bounding box
            box = create_bounding_box(x, y, z, dx, dy, dz, heading)
            bboxes.append(box)

# Visualize filtered point cloud and bounding boxes
# o3d.visualization.draw_geometries([pcd] + bboxes, window_name="Filtered Point Cloud with Bounding Boxes")
