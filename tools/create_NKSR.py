import faulthandler
faulthandler.enable()

import os
import sys
import torch
import numpy as np
import nksr
import open3d as o3d
from pycg import vis
from sklearn.neighbors import NearestNeighbors
from get_std import get_std

nksr_path = "submodules/NKSR/examples"
sys.path.insert(0, nksr_path)
std = None #0.025101243
alpha = 1

print("start script")
GPU = True

if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

data = np.load('./pcl_data.npz', allow_pickle=True)
pcl = data['array1'].reshape(-1, 3)
print(pcl)
inds = np.random.permutation(np.arange(pcl.shape[0]))[:pcl.shape[0]//12]
print("Point cloud shape:", pcl.shape)
#batched_pred = data['array3']
color = data['array2'] / 255.0
print("Color shape:", color.shape)

#camera_positions = np.vstack([pose["camera_positions_pointwise"] for pose in batched_pred])
#print("Camera positions shape:", camera_positions.shape)
if std is None or not os.path.exists("normals.pt"):
    print("estimatating std")
    std, points, normals = get_std(data, normal_fn=nksr.get_estimate_normal_preprocess_fn(64, 90.0))
    input_xyz = torch.from_numpy(np.asarray(pcl)[inds]).float().to(device) * (0.1/(std * alpha))
    #sensor_data = torch.from_numpy(np.asarray(camera_positions)).float().to(device)
    color_data = torch.from_numpy(np.asarray(color)[inds]).float().to(device)
    
    #get normals
    print("populating normals")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcl)[inds])

    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn.fit(np.asarray(points))
    del points

    distances, indices = nn.kneighbors(np.asarray(pcd.points))

    normals_full = normals[indices[:, 0]].type(torch.float32).to(device)
    del normals, nn, pcd
else:
    normals_full = torch.load("normals.pt")
    input_xyz = torch.from_numpy(np.asarray(pcl)[inds]).float().to(device) * (0.1/(std * alpha))
    #sensor_data = torch.from_numpy(np.asarray(camera_positions)).float().to(device)
    color_data = torch.from_numpy(np.asarray(color)[inds]).float().to(device)
torch.save(normals_full, "normals.pt")


print("Normal shape:", normals_full.shape)

reconstructor = nksr.Reconstructor(device)
reconstructor.chunk_tmp_device = torch.device("cpu")
print("Start reconstruction")
if GPU:
    field = reconstructor.reconstruct(
        input_xyz, normal=normals_full, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        chunk_size=25, overlap_ratio=0.1,
        solver_max_iter=5000,
        #preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )
else:
    from recons_waymo_cpu import normal_func
    field = reconstructor.reconstruct(
        input_xyz, sensor=sensor_data, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        chunk_size=20,
        # solver_max_iter=5000,
        preprocess_fn=normal_func
    )
print("set colour")
# colour
field.set_texture_field(nksr.fields.PCNNField(input_xyz, color_data))

mesh = field.extract_dual_mesh(mise_iter=1)

vis_mesh = vis.mesh(mesh.v/(0.1/(std * alpha)), mesh.f, color=mesh.c)

file = "output_NKSR.ply"

mesh_path = os.path.join(file)
o3d.io.write_triangle_mesh(mesh_path, vis_mesh)
print("saved mesh file")