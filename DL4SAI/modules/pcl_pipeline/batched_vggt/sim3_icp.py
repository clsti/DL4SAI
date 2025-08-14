import os
import numpy as np
from transformation import Trf

class Sim3ICP:
    def __init__(self, pcls_path, verbose=False, mode='umeyama_weighted'):
        """
        
        """

        self.pcls_path = pcls_path
        self.verbose = verbose

        self.trf_initial = Trf(os.path.join(self.pcls_path, "initial"), verbose=verbose)
        self.trf_loc_align = Trf(os.path.join(self.pcls_path, "loc_align"), verbose=verbose)

        if mode not in ['umeyama_weighted', 'umeyama', 'scale_translation']:
            print(f"[Warning] Unsupported mode '{mode}'. Expected 'umeyama_weighted', 'umeyama', or 'scale_translation'. Using 'umeyama_weighted' as default.")
            self.mode = 'umeyama_weighted'
        else:
            self.mode = mode

    def run(self, pcl_transformed, pcl_transformed_filtered, batched_pred):
        pairwise_transforms = self.compute_pairwise_transforms(pcl_transformed, batched_pred)
        pcl = self.trf_loc_align.run(pcl_transformed_filtered, pairwise_transforms, batched_pred)

        return pcl
    
    def compute_pairwise_transforms(self, pcl_transformed, batched_pred):
        '''
        
        '''
        pairwise_transforms = {}
        n = len(pcl_transformed) - 1
        
        for i in range(n):
            j = i + 1
            if self.verbose:
                print(f"Aligning point cloud {j} with point cloud {i}...")

            # Extract point clouds, confidence values and masks
            tgt_raw = pcl_transformed[i]
            tgt_vid_sizes = batched_pred[i]["vid_img_sizes"]
            tgt_confidence_raw = batched_pred[i]["confidence"]
            tgt_conf_mask_align_raw = batched_pred[i]["conf_mask_align"]
            src_raw = pcl_transformed[j]
            src_vid_sizes = batched_pred[j]["vid_img_sizes"]
            src_confidence_raw = batched_pred[j]["confidence"]
            src_conf_mask_align_raw = batched_pred[j]["conf_mask_align"]

            # Extract overlapping point clouds and confidence values
            tgt = tgt_raw[tgt_vid_sizes[0]:].reshape(-1, 3)
            tgt_confidence = tgt_confidence_raw[tgt_vid_sizes[0]:].reshape(-1)
            tgt_conf_mask_align = tgt_conf_mask_align_raw[tgt_vid_sizes[0]:].reshape(-1)
            src = src_raw[:src_vid_sizes[0]].reshape(-1, 3)
            src_confidence = src_confidence_raw[:src_vid_sizes[0]].reshape(-1)
            src_conf_mask_align = src_conf_mask_align_raw[:src_vid_sizes[0]].reshape(-1)

            if self.verbose:
                print(f"tgt shape original: {tgt.shape}, src shape original: {src.shape}")

            # Filter point clouds based and confidences based on masks
            mask_align_combinded = tgt_conf_mask_align & src_conf_mask_align
            tgt = tgt[mask_align_combinded]
            tgt_confidence = tgt_confidence[mask_align_combinded]
            src = src[mask_align_combinded]
            src_confidence = src_confidence[mask_align_combinded]

            if self.verbose:
                print(f"tgt shape filtered: {tgt.shape}, src shape filtered: {src.shape}")

            if self.verbose:
                tgt_ply = batched_pred[i]["vertices"]
                src_ply = batched_pred[j]["vertices"]
                tgt_color = batched_pred[i]["colors"]
                src_color = batched_pred[j]["colors"]

                tgt_name = os.path.join(self.pcls_path, "initial", f"pcd_{j}.ply")
                src_name = os.path.join(self.pcls_path, "initial", f"pcd_{i}.ply")

                self.trf_initial.to_pcd_file(tgt_ply.reshape(-1, 3), tgt_color, tgt_name)
                self.trf_initial.to_pcd_file(src_ply.reshape(-1, 3), src_color, src_name)

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
