from . import mast3r_wrapper as m3
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from contextlib import nullcontext
import tempfile
import numpy as np
import open3d as o3d
import os
import copy
import torch
from scipy.spatial import cKDTree

class Scaling:

    def __init__(self, batched_images, device=None):
        """
        
        """
        weights_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

        self.model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        self.chkpt_tag = hash_md5(weights_path)
        self.device = device if device else torch.device("cpu")
        self.batched_images = batched_images

    def run_master(self, imgs):
        with tempfile.TemporaryDirectory(suffix='_mast3r') as tmpdirname:
            cache_path = os.path.join(tmpdirname, self.chkpt_tag)
            os.makedirs(cache_path, exist_ok=True)
            return m3.get_reconstructed_scene(cache_path,
                                       gradio_delete_cache=None,
                                       model=self.model,
                                       retrieval_model=None,
                                       device=self.device,
                                       silent=False,
                                       image_size=512,
                                       current_scene_state=None,
                                       filelist=imgs,
                                       optim_level='refine+depth',
                                       lr1=0.07,
                                       niter1=300,
                                       lr2=0.01,
                                       niter2=300,
                                       min_conf_thr=0,
                                       matching_conf_thr=0,
                                       as_pointcloud=True,
                                       mask_sky=False,
                                       clean_depth=True,
                                       transparent_cams=False,
                                       cam_size=0.2,
                                       scenegraph_type='complete',
                                       winsize=1,
                                       win_cyclic=False,
                                       refid=0,
                                       TSDF_thresh=0,
                                       shared_intrinsics=True)
            
    def get_scale(source, target):
        mean_source = np.mean(source, axis=0, keepdims=True)
        mean_target = np.mean(target, axis=0, keepdims=True)
        s1 = np.median(np.abs(source - mean_source))
        s2 = np.median(np.abs(target - mean_target))

        source = (source - mean_source) * (s2/s1) + mean_source

        s_final = 1
        t_final = 0
        
        target_kdtree = cKDTree(target)

        for _ in range(5):
            # Find closest points in target for each source point
            _, indices = target_kdtree.query(source, k=1)
            target_matched = target[indices]   # shape: (N_src, 3)

            source = source.T  # (3, N)
            target_matched = target_matched.T  # (3, N)
            N = source.shape[1]
        
            # Compute means
            mean_source = np.mean(source, axis=1, keepdims=True)
            mean_target = np.mean(target_matched, axis=1, keepdims=True)

            # Center the point sets
            source_centered = source - mean_source
            target_centered = target_matched - mean_target
            
            # Compute variance and covariance
            var = np.sum(source_centered ** 2) / N
            covariance = np.sum(target_centered * source_centered) / N
            
            # Compute scale
            s = covariance / var
            
            # Compute translation
            t = mean_target - s * mean_source

            source = source.T * s + t.T

            s_final *= s
            t_final += t # can be ignored

        return s_final*(s2/s1)

         
    def run(self, pcl) -> float:
        """
        Subdivides each batch into 3 sub-batches and calculates the variance in scale estimation for these sub-batches.
        Final scale is the mean scale for the batch with the smallest variance.

        returns: final_scale
        """
        results = []
        for i in range(len(self.batched_images)):
            images = self.batched_images[i]
            batches = [images[:len(images)//3], images[len(images)//3:2*len(images)//3], images[2*len(images)//3:]]
            batches = [batch[np.arange(0, len(batch), len(batch)//2)] for batch in batches]
            scales = []
            for batch in batches:
                target = self.run_master(batch)
                scales.append(self.get_scale(pcl[i], target)) #TODO: what is the format of pcl
            scales = np.array(scales)
            results.append([scales.mean(), scales.var()])
        
        results = np.array(results)

        return results[np.argmin(results[:, 1])][0]
            
        
if __name__ == "__main__":
    Scaling(None)
    print("ran")