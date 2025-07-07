import mast3r_wrapper as m3
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from contextlib import nullcontext
import tempfile
import numpy as np
import open3d as o3d
import os
import copy

class Scaling:

    def __init__(self, batched_images, device=None):
        """
        
        """
        weights_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

        self.model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
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
        source_pts = np.asarray(source.points)
        target_pts = np.asarray(target.points)
        scale_source = np.mean(np.linalg.norm(source_pts - np.mean(source_pts, axis=0), axis=1))
        scale_target = np.mean(np.linalg.norm(target_pts - np.mean(target_pts, axis=0), axis=1))
        
        scale = scale_target / scale_source
        print("scale:", scale)

        target_pcd = copy.deepcopy(target)
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        source_scaled = copy.deepcopy(source)
        source_scaled.scale(scale, center=source_scaled.get_center())

        # Step 2: Apply ICP for fine alignment
        threshold = 0.02  # Distance threshold for ICP
        trans_init = np.eye(4)

        # Align centroids
        source_center = source_scaled.get_center()
        target_center = target_pcd.get_center()
        translation = target_center - source_center

        # Apply translation to source
        source_scaled.translate(translation)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_scaled, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        print("ICP fitness:", reg_p2p.fitness)
        print("ICP transformation:\n", reg_p2p.transformation)

        #aligned = source_scaled.transform(reg_p2p.transformation)
        #o3d.io.write_point_cloud(target_dir + "/aligned.ply", aligned)

        return scale
         
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
            batches = [batch[np.arange(0, len(batch), len(batch)//6)] for batch in batches]
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