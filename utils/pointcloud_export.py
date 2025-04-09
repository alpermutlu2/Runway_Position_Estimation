
import numpy as np
import open3d as o3d
import os

def export_pointcloud(depth_map, pose, fx=100, fy=100, cx=None, cy=None, save_path='cloud.ply'):
    h, w = depth_map.shape
    if cx is None: cx = w / 2
    if cy is None: cy = h / 2

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    transformed = (pose @ xyz_hom.T).T[:, :3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(transformed)
    o3d.io.write_point_cloud(save_path, pc)
    print(f"Saved point cloud with {len(pc.points)} points to {save_path}")
