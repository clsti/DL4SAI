import os
import numpy as np
import open3d as o3d

class Sim3ICP:
    def __init__(self, pcls_path, verbose=False, mode='umeyama_weighted'):
        """
        
        """

        self.pcls_path = pcls_path
        self.verbose = verbose
        self.transformation_chain_to_world = []

        if self.verbose:
            os.makedirs(os.path.join(self.pcls_path, "initial"), exist_ok=True)
            os.makedirs(os.path.join(self.pcls_path, "trsf"), exist_ok=True)

        if mode not in ['umeyama_weighted', 'umeyama', 'scale_translation']:
            print(f"[Warning] Unsupported mode '{mode}'. Expected 'umeyama_weighted', 'umeyama', or 'scale_translation'. Using 'umeyama_weighted' as default.")
            self.mode = 'umeyama_weighted'
        else:
            self.mode = mode

    def run(self, pcl_transformed, pcl_transformed_filtered, batched_pred):
        pairwise_transforms = self.compute_pairwise_transforms(pcl_transformed, batched_pred)
        pcl = self.transform_pcls(pcl_transformed_filtered, pairwise_transforms, batched_pred)

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
            tgt_confidence_raw = batched_pred[i]["confidence"]
            src_raw = pcl_transformed[j]
            src_vid_sizes = batched_pred[j]["vid_img_sizes"]
            src_confidence_raw = batched_pred[j]["confidence"]

            tgt = tgt_raw[tgt_vid_sizes[1]:].reshape(-1, 3)
            tgt_confidence = tgt_confidence_raw[tgt_vid_sizes[1]:].reshape(-1)
            src = src_raw[:src_vid_sizes[0]].reshape(-1, 3)
            src_confidence = src_confidence_raw[:src_vid_sizes[0]].reshape(-1)

            if self.verbose:
                tgt_ply = batched_pred[i]["vertices"]
                src_ply = batched_pred[j]["vertices"]
                tgt_color = batched_pred[i]["colors"]
                src_color = batched_pred[j]["colors"]

                tgt_name = os.path.join(self.pcls_path, "initial", f"pcd_{j}.ply")
                src_name = os.path.join(self.pcls_path, "initial", f"pcd_{i}.ply")

                self.to_pcd_file(tgt_ply.reshape(-1, 3), tgt_color, tgt_name)
                self.to_pcd_file(src_ply.reshape(-1, 3), src_color, src_name)

            if self.mode == 'umeyama_weighted':
                s, R, t = self.umeyama(src, tgt, src_confidence, tgt_confidence, with_scale=True)
            elif self.mode == 'umeyama':
                s, R, t = self.umeyama(src, tgt)
            elif self.mode == 'scale_translation':
                s, R, t = self.align_point_clouds_scale_translation(src, tgt)
            else:
                print(f"Unknown mode: {self.mode}. Using 'umeyama_weighted' as default.")
                s, R, t = self.umeyama(src, tgt, src_confidence, tgt_confidence, with_scale=True)

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

    def umeyama(self, src, tgt, src_weights=None, tgt_weights=None, with_scale=True):
        """
        (Weighted) Umeyama algorithm.

        Args:
            src: (N, d) numpy array of source points
            tgt: (N, d) numpy array of target points
            src_weights: (N,) array of source weights
            tgt_weights: (N,) array of target weights
            with_scale: bool, if True estimate scale, else assume scale=1

        Returns:
            s: scale (scalar)
            R: rotation matrix (3 x 3)
            t: translation vector (3,)
        """
        assert src.shape == tgt.shape, "Input point sets must have the same shape."
        n, d = src.shape

        weights = None
        if src_weights is not None and tgt_weights is not None:
            assert src.shape[0] == src_weights.shape[0]
            assert tgt.shape[0] == tgt_weights.shape[0]

            weights = np.sqrt(src_weights * tgt_weights)

        # 1. Compute centroids
        if weights is not None:
            src_mean = np.average(src, axis=0, weights=weights)
            tgt_mean = np.average(tgt, axis=0, weights=weights)
        else:
            # Unweighted mean
            src_mean = src.mean(axis=0)
            tgt_mean = tgt.mean(axis=0)

        # 2. Center point clouds
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        # 3. Compute covariance matrix
        if weights is not None:
            W = np.sum(weights)
            Sigma = (tgt_centered.T * weights) @ src_centered / W
        else:
            # unweighted covariance
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
            if weights is not None:
                squared_norms = np.sum(src_centered**2, axis=1)
                var_src = np.sum(weights * squared_norms) / W
            else:
                var_src = np.sum(src_centered ** 2) / n

            s = np.sum(D * np.diag(S)) / var_src
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