import os
import shutil
import torch
import numpy as np
import pypose as pp
import open3d as o3d

class Trf:
    def __init__(self, pcl_path, verbose=False):
        """
        
        """
        self.pcl_path = pcl_path
        self.verbose = verbose
        self.transformation_chain_to_world = []

        if self.verbose:
            if not os.path.exists(self.pcl_path):
                os.makedirs(self.pcl_path)

    def run(self, pcl_list, pairwise_transforms, batched_pred):
        self.path_cleanup()
        return self.transform_pcls(pcl_list, pairwise_transforms, batched_pred)


    def path_cleanup(self):
        """
        
        """
        if self.verbose:
            if os.path.exists(self.pcl_path):
                shutil.rmtree(self.pcl_path)
            os.makedirs(self.pcl_path)

    def to_pcd_file(self, pcl, color, path):
        """
        
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color/255.0)
        o3d.io.write_point_cloud(path, pcd, write_ascii=False)

    def to_transformation_matrix(self, s, R, t):
        """
        Build a transformation matrix from scale, rotation, and translation.
        """
        H = np.eye(4)
        H[:3, :3] = s * R
        H[:3, 3] = t
        return H
    
    def sim3_transformation(self, points, H_sim3):
        """
        Apply a Sim(3) transformation (scale, rotation, translation) to a batch of (n, h, w, 3) points.
        """
        orig_shape = points.shape
        n_points = np.prod(orig_shape[:-1])
        points_flat = points.reshape(-1, 3)

        points_h = np.hstack((points_flat, np.ones((n_points, 1))))
        # ((4,4) @ (n,4).T).T = (n,4)
        points_trans_h = (H_sim3 @ points_h.T).T 
        points_trans = points_trans_h[:, :3].reshape(orig_shape)
        return points_trans
    
    def sim3_transformation_extrinsics(self, extrinsics, H_sim3):
        """
        Apply a Sim(3) transformation to a batch of extrinsics matrices (n, 3,4).
        """
        n_extrinsics = extrinsics.shape[0]

        extrinsics_h = np.concatenate([
            extrinsics,
            np.tile(np.array([0, 0, 0, 1])[None, None, :], (n_extrinsics, 1, 1))  # (n, 1, 4)
        ], axis=1)

        H_torch = torch.from_numpy(H_sim3).float().unsqueeze(0)
        sim3 = pp.from_matrix(H_torch, ltype=pp.Sim3_type)

        sim3_inv = sim3.Inv()
        H_inv = sim3_inv.matrix().squeeze(0).numpy()

        extrinsics_trans = extrinsics_h @ H_inv
        extrinsics_out = extrinsics_trans[:, :3, :]  # (n, 3, 4)

        return extrinsics_out
    
    def transform_pcls(self, pcl_list, pairwise_transforms, batched_pred):
        """
        Transform all point clouds into the frame of pcl_list[0].

        Args:
            pcl_list: list of N point clouds as (N, H, W, 3) np.ndarrays
            pairwise_transforms: dict of {(i, j): (s, R, t)} representing transforms from pcl[i] to pcl[j]

        Returns:
            List of aligned point clouds in the same frame
        """
        pcl_transformed = [pcl_list[0].copy()]
        cumulative_transform = np.eye(4)

        if self.verbose:
            path = os.path.join(self.pcl_path, f"pcd_0.ply")
            self.to_pcd_file(pcl_list[0].reshape(-1, 3), batched_pred[0]["colors"], path)

        for (i, j), (s, R, t) in pairwise_transforms.items():
            if self.verbose:
                self.transformation_chain_to_world.append(cumulative_transform)

            H = self.to_transformation_matrix(s, R, t)
            
            # Update cumulative transformation
            cumulative_transform = cumulative_transform @ H

            if self.verbose:
                self.transformation_chain_to_world.append(cumulative_transform)

            pcl_trans = self.sim3_transformation(pcl_list[j], cumulative_transform)
            batched_pred[j]["extrinsics"] = self.sim3_transformation_extrinsics(batched_pred[j]["extrinsics"], cumulative_transform)
            batched_pred[j]["camera_positions_pointwise"] = self.sim3_transformation(batched_pred[j]["camera_positions_pointwise"], cumulative_transform)

            if self.verbose:
                color = batched_pred[j]["colors"]
                tgt_name = os.path.join(self.pcl_path, f"pcd_{j}.ply")
                self.to_pcd_file(pcl_trans.reshape(-1, 3), color, tgt_name)

            pcl_transformed.append(pcl_trans)

        return pcl_transformed