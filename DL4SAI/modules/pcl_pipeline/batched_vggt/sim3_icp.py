import os
import numpy as np
import open3d as o3d

class Sim3ICP:
    def __init__(self, pcls_path, verbose=False, correct_rotation=True):
        """
        
        """

        self.pcls_path = pcls_path
        self.verbose = verbose
        self.correct_rotation = correct_rotation
        self.transformation_chain_to_world = []

        if self.verbose:
            os.makedirs(os.path.join(self.pcls_path, "initial"), exist_ok=True)
            os.makedirs(os.path.join(self.pcls_path, "trsf"), exist_ok=True)


    def run(self, pcl_transformed, batched_pred):
        pairwise_transforms = self.compute_pairwise_transforms(pcl_transformed, batched_pred)
        pcl = self.transform_pcls(pcl_transformed, pairwise_transforms, batched_pred)

        return pcl
    
    def compute_pairwise_transforms(self, pcl_transformed, batched_pred):
        '''
        
        '''
        pairwise_transforms = {}
        n = len(pcl_transformed) - 1
        
        for i in range(n):
            j = i + 1
            tgt_raw = pcl_transformed[i]
            tgt_vid_sizes = batched_pred[i]["vid_img_sizes"]
            src_raw = pcl_transformed[j]
            src_vid_sizes = batched_pred[j]["vid_img_sizes"]

            tgt = tgt_raw[tgt_vid_sizes[1]:].reshape(-1, 3)
            src= src_raw[:src_vid_sizes[0]].reshape(-1, 3)

            if self.verbose:
                tgt_color = batched_pred[i]["colors"]
                src_color = batched_pred[j]["colors"]

                tgt_name = os.path.join(self.pcls_path, "initial", f"pcd_{j}.ply")
                src_name = os.path.join(self.pcls_path, "initial", f"pcd_{i}.ply")

                self.to_pcd_file(tgt_raw.reshape(-1, 3), tgt_color, tgt_name)
                self.to_pcd_file(src_raw.reshape(-1, 3), src_color, src_name)

            if self.correct_rotation:
                s, R, t = self.umeyama(src, tgt)
            else:
                s, R, t = self.align_point_clouds_scale_translation(src, tgt)

            pairwise_transforms[(i, j)] = (s, R, t)

        return pairwise_transforms

    def align_point_clouds_scale_translation(self, src, tgt):
        """
        Aligns point cloud src to point cloud tgt using only scale and translation.

        Args:
            src: (N, d) numpy array of source points
            tgt: (N, d) numpy array of target points

        Returns:
            s: optimal scale (scalar)
            R: Identity matrix (3 x 3)
            t: optimal translation (3,)
        """
        # 1. Compute centroids
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        
        # 2. Center point clouds
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        # Compute optimal scale using einsum for clarity and performance
        numerator = np.einsum('ij,ij->', tgt_centered, src_centered)
        denominator = np.einsum('ij,ij->', src_centered, src_centered)
        s = numerator / denominator

        # Compute optimal translation
        t = tgt_mean - s * src_mean

        return s, np.eye(3), t

    def umeyama(self, src, tgt, with_scale=True):
        """
        Umeyama algorithm for similarity (rigid + scale) transformation.

        Args:
            src: (N, d) numpy array of source points
            tgt: (N, d) numpy array of target points
            with_scale: bool, if True estimate scale, else assume scale=1

        Returns:
            s: scale (scalar)
            R: rotation matrix (3 x 3)
            t: translation vector (3,)
        """
        assert src.shape == tgt.shape, "Input point sets must have the same shape."
        n, d = src.shape

        # 1. Compute centroids
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)

        # 2. Center point clouds
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        # 3. Compute covariance matrix
        Sigma = (tgt_centered.T @ src_centered) / n

        # 4. SVD
        U, D, Vt = np.linalg.svd(Sigma)
        
        # 5. Compute reflection matrix to ensure a proper rotation (det=1)
        S = np.eye(d)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1

        # 6. Rotation
        R = U @ S @ Vt

        # 7. Scale
        if with_scale:
            var_P = np.sum(src_centered ** 2) / n
            s = np.sum(D * np.diag(S)) / var_P
        else:
            s = 1.0

        # 8. Translation
        t = tgt_mean - s * R @ src_mean

        return s, R, t

    def to_pcd_file(self, pcl, color, path):
        """
        
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.colors = o3d.utility.Vector3dVector(color/255.0)
        o3d.io.write_point_cloud(path, pcd)

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
            path = os.path.join(self.pcls_path, "trsf", f"pcd_0.ply")
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

            if self.verbose:
                color = batched_pred[j]["colors"]
                tgt_name = os.path.join(self.pcls_path, "trsf", f"pcd_{j}.ply")
                self.to_pcd_file(pcl_trans.reshape(-1, 3), color, tgt_name)

            pcl_transformed.append(pcl_trans)

        return pcl_transformed