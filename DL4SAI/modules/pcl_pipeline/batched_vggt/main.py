import os
import pickle
import numpy as np
import torch
import gc
from pymlg import SE3

from batching import Batching
from vggt_proc import VGGTproc
from merging import Merging
from scaling.main import Scaling
from sim3_icp import Sim3ICP
from glob_align import GlobAlign


class BatchedVGGT:
    """
    Batched VGGT processing
    """

    def __init__(self, 
                 data_path,
                 verbose=False,
                 use_cached_batches=False,
                 use_cached_pcls=False,
                 max_image_size=80,
                 conf_thres_visu=0.5,
                 conf_thres_align=0.7,
                 color=True,
                 mode='concatenate',
                 trf_mode='SE3',
                 image_path=None,
                 transition_filter={},
                 loop_closure=[]):
        """
        
        """

        self.verbose = verbose
        self.use_cached_pcls = use_cached_pcls
        self.trf_mode = trf_mode
        self.transition_filter = transition_filter
        self.conf_thres_visu = conf_thres_visu
        self.loop_closure = loop_closure

        if trf_mode not in ['SE3', 'rotation']:
            print(f"[Warning] Unsupported trf_mode '{trf_mode}'. Expected 'SE3' or 'rotation' (default: SE3).")

        if image_path is None:
            self.data_path = os.path.join(data_path,'generated_data')
        else:
            self.data_path = image_path

        self.image_path = os.path.join(self.data_path, 'images')
        self.pcls_path = os.path.join(self.data_path, 'pcls')
        self.cache_path = os.path.join(self.image_path, 'pcl_cache.pkl')
        self.target_file_path = os.path.join(self.data_path, 'glbs')
        os.makedirs(self.target_file_path, exist_ok=True)

        self.batching = Batching(data_path, self.image_path, verbose=verbose, use_cached=use_cached_batches, max_image_size=max_image_size)
        self.vggt_proc = VGGTproc(verbose=verbose, conf_thres_visu=conf_thres_visu, conf_thres_align=conf_thres_align)
        self.merging = Merging(mode=mode, verbose=verbose, color=color)
        self.loc_align = Sim3ICP(self.pcls_path, verbose=verbose, mode='umeyama_weighted', loop_closure=self.loop_closure)
        self.glob_align = GlobAlign(self.pcls_path, verbose=verbose)

        # Scaling
        self.scaling = Scaling(self.batches)

        # Get batches
        self.batches = self.batching.get_batches()
        self.batches_size = self.batching.get_batches_size()

        self.batched_pred = []

        self.transformation_chain = []
        self.pcl_transformed = []  # shape: list of (n, h, w, 3)
        self.pcl_transformed_filtered = []  # shape: list of (n, h, w, 3)
        self.pcl_trf_align = []  # shape: list of (n, h, w, 3)
        self.pcl_glob_align = []  # shape: list of (n, h, w, 3)
        self.pcl_glob_align_scaled = []  # shape: list of (n, h, w, 3)
        self.transformation_chain_to_world = []
        self.pairwise_transformation = []

        self.weight_flag = 2.0 # weight or None for no weight

    def run(self):
        """
        
        """
        if not self.use_cached_pcls:
            self.batched_predictions()
            self.create_trans_chain()
            self.transform_pcls()
            self.local_alignment()
            self._cache_data()
            self.unload()
        else:
            self._load_cache()
            print("Loaded pcls from cache")

        self.global_alignment()

        self.apply_scaling()
        pcl, colors = self.merge()
        return self.batched_pred, pcl, colors


    def batched_predictions(self):
        """
        
        """
        # process batches
        for i, images in enumerate(self.batches):
            # filter for transitions
            if i in self.transition_filter:
                self.vggt_proc.set_conf_thres_visu(self.transition_filter[i])
            else:
                self.vggt_proc.set_conf_thres_visu(self.conf_thres_visu)

            # run vggt
            vertices_raw, vertices, colors, conf, extrinsics, intrinsics, conf_mask_align, cam_pos_pointwise = self.vggt_proc.run(images)
            predictions = {
                "vertices_raw": vertices_raw,
                "vid_img_sizes": self.batches_size[i],
                "vertices": vertices,
                "colors": colors,
                "confidence": conf,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "conf_mask_align": conf_mask_align,
                "camera_positions_pointwise": cam_pos_pointwise,
            }
            self.batched_pred.append(predictions)

    def to_SE3(self, H=None):
        """
        Constructs an SE(3) transformation matrix.
        """
        if H is None:
            rotation_matrix = np.eye(3)
            translation_vector = np.zeros(3)
        else:
            assert H.shape == (3, 4), "Input H must be a 3x4 matrix [R | t]"
            rotation_matrix = H[:,:3]
            translation_vector = H[:,3]

        return SE3.from_components(rotation_matrix, translation_vector)

    def get_average(self, extr_curr, extr_next):
        assert len(extr_curr) == len(extr_next), "Batch size mismatch! "

        SE3_curr = [self.to_SE3(H) for H in extr_curr]
        SE3_next = [self.to_SE3(H) for H in extr_next]

        SE3_trans = [SE3.inverse(H_curr) @ H_next for H_curr, H_next in zip(SE3_curr, SE3_next)]
        Xi_trans = [SE3.log(H) for H in SE3_trans]

        if self.weight_flag is not None:
            weights = np.ones(len(extr_curr))
            weights[0] = self.weight_flag
            Xi_average = np.average(Xi_trans, axis=0, weights=weights)
        else:
            Xi_average = np.mean(Xi_trans, axis=0)

        return SE3.exp(Xi_average)

    def create_trans_chain(self):
        """
        
        """
        self.transformation_chain.append(self.to_SE3())

        for i in range(len(self.batches) - 1):
            extr_curr = self.batched_pred[i]["extrinsics"]
            extr_next = self.batched_pred[i+1]["extrinsics"]
            size_curr = self.batches_size[i]
            size_next = self.batches_size[i+1]

            extr_curr_batch = extr_curr[size_curr[0]:]
            extr_next_batch = extr_next[:size_next[0]]

            H = self.get_average(extr_curr_batch, extr_next_batch)
            self.transformation_chain.append(H)

    def SE3transform_pcl(self, H, points):
        """
        Apply SE(3) matrix to batch of (n, h, w, 3) points
        """
        orig_shape = points.shape
        n_points = np.prod(orig_shape[:-1])
        points_flat = points.reshape(-1, 3)

        points_h = np.hstack((points_flat, np.ones((n_points, 1))))
        # ((4,4) @ (n,4).T).T = (n,4)
        points_trans_h = (H @ points_h.T).T 
        points_trans = points_trans_h[:, :3].reshape(orig_shape)
        return points_trans

    def Rtransform_pcl(self, H, points):
        """
        Apply Rotation matrix to batch of (n, h, w, 3) points
        """
        orig_shape = points.shape
        points_flat = points.reshape(-1, 3)

        R = H[:3, :3]
        points_rotated = (R @ points_flat.T).T
        return points_rotated.reshape(orig_shape)
    
    def SE3transform_extrinsics(self, H, extrinsics):
        """
        
        """
        extr_new = []
        for extr in extrinsics:
            extr_inv = SE3.inverse(self.to_SE3(extr))
            extr_new.append(SE3.inverse(H @ extr_inv))

        extr_new = np.stack(extr_new, axis=0)
        extr_new = extr_new[:, :3, :]
            
        return extr_new


    def transform_pcls(self):
        """
        Transform all point clouds into the same coordinate frame
        """
        cumulative_transform = self.transformation_chain[0]
        self.transformation_chain_to_world.append(cumulative_transform)
        for i, pred in enumerate(self.batched_pred):
            pcl = pred["vertices_raw"]
            pcl_filtered = pred["vertices"]
            if i == 0:
                self.pcl_transformed.append(pcl)
                self.pcl_transformed_filtered.append(pcl_filtered)
            else:
                # Update cumulative transformation
                cumulative_transform = cumulative_transform @ self.transformation_chain[i]
                self.transformation_chain_to_world.append(cumulative_transform)

                if self.trf_mode == 'rotation':
                    pcl_trans = self.Rtransform_pcl(cumulative_transform, pcl)
                    pcl_trans_filtered = self.Rtransform_pcl(cumulative_transform, pcl_filtered)
                else:
                    pcl_trans = self.SE3transform_pcl(cumulative_transform, pcl)
                    pcl_trans_filtered = self.SE3transform_pcl(cumulative_transform, pcl_filtered)

                pred["extrinsics"] = self.SE3transform_extrinsics(cumulative_transform, pred["extrinsics"])
                self.pcl_transformed.append(pcl_trans)
                self.pcl_transformed_filtered.append(pcl_trans_filtered)

    def _cache_data(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.batched_pred, self.pcl_transformed, self.pcl_transformed_filtered, self.transformation_chain, self.transformation_chain_to_world, self.pcl_trf_align, self.pairwise_transformation), f)
        except Exception as e:
            if self.verbose:
                print(f"Failed to save pcl cache: {e}")

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            if self.verbose:
                print("Pcl cache file not found.")
            return False

        try:
            with open(self.cache_path, 'rb') as f:
                self.batched_pred, self.pcl_transformed, self.pcl_transformed_filtered, self.transformation_chain, self.transformation_chain_to_world, self.pcl_trf_align, self.pairwise_transformation = pickle.load(f)
            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to load pcl cache: {e}")
            return False

    def unload(self):
        """Free VGGT from GPU memory."""
        if self.vggt_proc.model is not None:
            self.vggt_proc.model.to("cpu")
            del self.vggt_proc.model
            self.vggt_proc.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def local_alignment(self):
        self.pcl_trf_align, self.pairwise_transformation, self.loop_closure = self.loc_align.run(self.pcl_transformed, self.pcl_transformed_filtered, self.batched_pred)

    def global_alignment(self):
        if len(self.loop_closure) > 0:
            self.pcl_glob_align = self.glob_align.run(self.loop_closure, self.batched_pred)
        else:
            print("No loop constraints provided. Point clouds only locally aligned.")
            self.pcl_glob_align = self.pcl_trf_align

    def apply_scaling(self):
        scaling = self.scaling.run(self.pcl_glob_align)

        self.pcl_glob_align_scaled = [pcl * scaling for pcl in self.pcl_glob_align]

    def merge(self):
        return self.merging.run(self.pcl_glob_align_scaled, self.batched_pred)
