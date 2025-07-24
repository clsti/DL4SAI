import sys
import os
import torch
import numpy as np
import matplotlib
import trimesh

# project_root = os.path.join("../../../../")
# vggt_path = os.path.join(project_root, "submodules/vggt")
vggt_path = os.path.join("submodules/vggt")
sys.path.insert(0, vggt_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import integrate_camera_into_scene, apply_scene_alignment

class VGGTproc:
    """
    VGGT image processing
    """

    def __init__(self, verbose=False, pcl_pred="from_depth", conf_thres_visu=0.5, conf_thres_align=0.7):
        """
        load model
        """
        if pcl_pred not in ["from_depth", "3D_points"]:
            raise ValueError("Mode must be either 'from_depth' or '3D_points'")
        
        self.verbose = verbose
        self.pcl_pred = pcl_pred
        self.conf_thres_visu = conf_thres_visu
        self.conf_thres_align = conf_thres_align
        self.show_cam = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

        self.predictions = None
        self.vertices_raw = None
        self.vertices_3d = None
        self.camera_extrinsics = None
        self.camera_positions_pointwise = None
        self.intrinsics = None
        self.colors_rgb = None
        self.conf_mask_align = None

    def run(self, image_paths, target_file=None):
        self._proc(image_paths)
        self._post_proc(target_file)

        return self.vertices_raw, self.vertices_3d, self.colors_rgb, self.conf, self.camera_extrinsics, self.intrinsics, self.conf_mask_align, self.camera_positions_pointwise
    
    def set_conf_thres_visu(self, conf):
        self.conf_thres_visu = conf

    def _proc(self, image_paths):
        # preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)

        # run model
        with torch.no_grad():
            with torch.amp.autocast("cuda",dtype=self.dtype):
                self.predictions = self.model(images)

        # extract extrinsics/intrinsics
        extrinsic, intrinsic = pose_encoding_to_extri_intri(self.predictions["pose_enc"], images.shape[-2:])
        self.predictions["extrinsic"] = extrinsic
        self.predictions["intrinsic"] = intrinsic

        for key in self.predictions.keys():
            if isinstance(self.predictions[key], torch.Tensor):
                self.predictions[key] = self.predictions[key].cpu().numpy().squeeze(0)

        # Generate 3D points from depth map & camera parameters (more accurate than directly using point map)
        depth_map = self.predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, self.predictions["extrinsic"], self.predictions["intrinsic"])
        self.predictions["world_points_from_depth"] = world_points
    
    def _post_proc(self, target_file=None):
        if "3D_points" in self.pcl_pred:
            if "world_points" in self.predictions:
                pred_world_points = self.predictions["world_points"]  # No batch dimension to remove
                pred_world_points_conf = self.predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
            else:
                print("Warning: world_points not found in predictions, falling back to depth-based points")
                pred_world_points = self.predictions["world_points_from_depth"]
                pred_world_points_conf = self.predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Using Depthmap and Camera Branch")
            pred_world_points = self.predictions["world_points_from_depth"]
            pred_world_points_conf = self.predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

        # Get images from predictions
        images = self.predictions["images"]
        # Use extrinsic matrices instead of pred_extrinsic_list
        camera_matrices = self.predictions["extrinsic"]
        
        # store extrinsics & intrinsics
        self.camera_extrinsics = camera_matrices
        self.intrinsics = self.predictions["intrinsic"]
        
        pred_world_points_shape = pred_world_points.shape
        self.camera_positions_pointwise = np.expand_dims(np.expand_dims(camera_matrices[:,:,3], axis=1), axis=2)
        self.camera_positions_pointwise = np.tile(self.camera_positions_pointwise, (1, pred_world_points_shape[1], pred_world_points_shape[2], 1))

        # TODO: add mask_sky (maybe also mask_black & mask_white)

        self.vertices_raw = pred_world_points
        self.vertices_3d = pred_world_points.reshape(-1, 3)
        self.camera_positions_pointwise = self.camera_positions_pointwise.reshape(-1, 3)
        self.conf = pred_world_points_conf

        # Handle different image formats - check if images need transposing
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            self.colors_rgb = np.transpose(images, (0, 2, 3, 1))
        else:  # Assume already in NHWC format
            self.colors_rgb = images
        self.colors_rgb = (self.colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

        conf = pred_world_points_conf.reshape(-1)
        # Convert percentage threshold to actual confidence value for visualization
        if self.conf_thres_visu == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, self.conf_thres_visu * 100.0)

        # Convert percentage threshold to actual confidence value for alignment
        if self.conf_thres_align == 0.0:
            conf_threshold_align = 0.0
        else:
            conf_threshold_align = np.percentile(conf, self.conf_thres_align * 100.0)

        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
        self.conf_mask_align = (self.conf >= conf_threshold_align) & (self.conf > 1e-5)

        self.vertices_3d = self.vertices_3d[conf_mask]
        self.camera_positions_pointwise = self.camera_positions_pointwise[conf_mask]
        self.colors_rgb = self.colors_rgb[conf_mask]

        # TODO: where to store glb file? if verbose

        if self.verbose and target_file is not None:
            if self.vertices_3d is None or np.asarray(self.vertices_3d).size == 0:
                self.vertices_3d = np.array([[1, 0, 0]])
                self.colors_rgb = np.array([[255, 255, 255]])
                scene_scale = 1
            else:
                # Calculate the 5th and 95th percentiles along each axis
                lower_percentile = np.percentile(self.vertices_3d, 5, axis=0)
                upper_percentile = np.percentile(self.vertices_3d, 95, axis=0)

                # Calculate the diagonal length of the percentile bounding box
                scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

            colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

            # Initialize a 3D scene
            scene_3d = trimesh.Scene()

            # Add point cloud data to the scene
            point_cloud_data = trimesh.PointCloud(vertices=self.vertices_3d, colors=self.colors_rgb)

            scene_3d.add_geometry(point_cloud_data)

            # Prepare 4x4 matrices for camera extrinsics
            num_cameras = len(camera_matrices)
            extrinsics_matrices = np.zeros((num_cameras, 4, 4))
            extrinsics_matrices[:, :3, :4] = camera_matrices
            extrinsics_matrices[:, 3, 3] = 1

            if self.show_cam:
                # Add camera models to the scene
                for i in range(num_cameras):
                    world_to_camera = extrinsics_matrices[i]
                    camera_to_world = np.linalg.inv(world_to_camera)
                    rgba_color = colormap(i / num_cameras)
                    current_color = tuple(int(255 * x) for x in rgba_color[:3])

                    integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

            # Align scene to the observation of the first camera
            scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

            scene_3d.export(file_obj=target_file)
