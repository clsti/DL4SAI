import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import torch

def get_std(data, normal_fn, device="cuda"):
    mean_tr = 5
    k_norm = 50
    k_surf = 50
    
    batched_pred = data['array3']
    camera_positions = np.vstack([pose["camera_positions_pointwise"] for pose in batched_pred])
    sensor_data = torch.from_numpy(np.asarray(camera_positions)).float().to(device)
    
    # Assuming points is (N, 3) numpy array
    pcd = o3d.geometry.PointCloud()
    inds_full = np.random.permutation(np.arange(data['array1'].shape[0]))[:data['array1'].shape[0]//100]

    inds_full = inds_full[:-(inds_full.shape[0]%100)].reshape(100, -1)
    indices_full = []
    stds = []
    normals_full = []
    normals_ret = []
    for i, inds in enumerate(inds_full):
        x, normals, k_indices = normal_fn(torch.tensor(data['array1'])[inds].float().to(device), None, sensor_data[inds])
        #pcd.points = o3d.utility.Vector3dVector(data['array1'][inds])

        # Estimate normals (uses nearest neighbors)
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_norm))
        # Optional: Orient normals consistently towards a camera location
        #pcd.orient_normals_consistent_tangent_plane(200)  # larger = more global consistency

        #nn = NearestNeighbors(n_neighbors=k_surf, algorithm='kd_tree', n_jobs=-1)
        #nn.fit(np.asarray(pcd.points))

        #distances, k_indices = nn.kneighbors(np.asarray(pcd.points))

        #x = torch.tensor(np.asarray(pcd.points)).to(device)
        #normals = torch.tensor(np.asarray(pcd.normals)).unsqueeze(1).to(device)
        k_norms = normals[k_indices].squeeze().to(device)
        projections = torch.sum(k_norms * normals.unsqueeze(1), dim=-1, keepdim=True).squeeze()  # (100000, 30, 1)
        means = projections.mean(dim=1)
        indices = np.argwhere(np.degrees(np.arccos(means.cpu()))<mean_tr).squeeze()
        normals_full.append(normals[indices])
        normals_ret.append(normals)
        
        del k_norms, projections, means
        
        devs = (x[k_indices] - x.unsqueeze(1)).to(device)
        dot_products = torch.sum(devs * normals.unsqueeze(1), dim=-1, keepdim=True).squeeze()  # (100000, 30, 1)
        indices_full.append(torch.tensor(inds[indices]))
        std = dot_products[indices].std(dim=1).mean().cpu()
        stds.append(std)
        print("estimated batch", i)

    std = np.mean(stds)
    print(std)
    indices_full = torch.concatenate(indices_full)
    normals_full = torch.concatenate(normals_full)
    x = torch.tensor(data['array1'])
    # Start points (e.g., point cloud centroids or any origin points)
    origins = x[indices_full].cpu()         # shape (N, 3)

    # Vector directions (e.g., normals, or any 3D vectors)
    directions = normals_full.cpu()     # shape (N, 3)

    # Length scale factor
    scale = std
    ends = origins + scale * directions

    # Create line segments between origins and ends
    points_for_lines = np.vstack((origins, ends))  # shape (2N, 3)
    lines = [[i, i + len(origins)] for i in range(len(origins))]

    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_for_lines)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optional: color the lines
    colors = [[1, 0, 0] for _ in lines]  # Red lines
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Save point cloud
    pcd.points = o3d.utility.Vector3dVector(x[np.concatenate(inds_full)])
    pcd.colors = o3d.utility.Vector3dVector(data['array2'][np.concatenate(inds_full)]/255)
    o3d.io.write_point_cloud("std_pointcloud.ply", pcd)

    # Save line set (e.g., vectors)
    o3d.io.write_line_set("std_normal.ply", line_set)

    return std, x[np.concatenate(inds_full)], torch.concatenate(normals_ret)

if __name__ == "__main__":
    get_std(data=np.load('./local_ws/MI_front_area/pcl_data.npz', allow_pickle=True))