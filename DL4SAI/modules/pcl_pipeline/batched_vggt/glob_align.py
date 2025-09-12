import os
import torch
import time
import numpy as np
import pypose as pp
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as SciRot
from transformation import Trf
import matplotlib.pyplot as plt


class GlobAlign:
    def __init__(self, pcls_path, verbose=False):
        
        self.pcls_path = os.path.join(pcls_path, "glob_align")
        self.verbose = verbose
        
        self.trf_glob_align = Trf(self.pcls_path, verbose=verbose)

    def run(self, loop_closure, batched_pred):
        input_poses = self.create_transformations(batched_pred)
        # torch.save({"poses": input_poses}, "data_input_poses.pt")
        H_abs_opt = self.global_optimization(input_poses, loop_closure)

        self.plot_extrinsics(input_poses, H_abs_opt)

        pcl = self.transform(H_abs_opt, batched_pred)

        return pcl

    def create_transformations(self, batched_pred):
        """

        """
        H_abs_pp = []

        for pred in batched_pred:
            H = np.eye(4)
            H[:3, :] = pred["extrinsics"][0]
            H_torch = torch.from_numpy(H).float()
            H_sim3 = pp.from_matrix(H_torch, ltype=pp.Sim3_type)
            H_abs_pp.append(H_sim3.Inv())

        return torch.stack(H_abs_pp)

    def global_optimization(self, input_poses, loop_closure=None):
        """
        
        """
        max_iterations = 200
        lambda_init = 1e-6
        self.solve_system_version = 'python'

        loop_closure = loop_closure if loop_closure is not None else []

        # Create loop closure constraints
        ii_loop, jj_loop, dSloop = self.create_constraints(loop_closure)

        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []

        # H_abs_init = input_poses.clone()

        # L-M loop
        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
            
            # num_seq = len(input_poses) - 1
            # num_lc = dSloop.shape[0]

            # resid_seq = resid[:num_seq]
            # resid_lc  = resid[num_seq:]

            # loop_error_norms = torch.linalg.norm(resid_lc, dim=1)
            # print("Loop closure errors:", loop_error_norms)

            
            if resid.numel() == 0:
                print("No residuals to optimize")
                break

            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)
            
            try: # Solve linear system
                begin_time = time.time()
                if self.solve_system_version == 'cpp':
                    delta_pose, = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                elif self.solve_system_version == 'python':
                    delta_pose = self.solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
                else:
                    print(f"Solver version has not been chosen! ('python' or 'cpp')")
                end_time = time.time()
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break
            
            Ginv_tmp = Ginv + delta_pose
            
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else float('inf')
            
            # L-M
            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
                print(f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (accepted)", end=' | ')

                # Convert to absolute poses
                Gopt = pp.Exp(Ginv).Inv()
                H_rel_opt = self.absolute_to_sequential_poses(Gopt)
                H_abs_opt = self.sequential_to_absolute_poses(H_rel_opt)

                # Plot extrinsics
                # save_path = os.path.join(f"extrinsics_step_{itr}.png")
                # self.plot_extrinsics(H_abs_init, H_abs_opt, save_path=save_path)
            else:
                lmbda *= 2
                print(f"Iteration {itr}: cost {current_cost:.14f} -> {new_cost:.14f} (rej)     ", end=' | ') # more readible to accepted
            
            print(f'Time of solver ({self.solve_system_version}): {(end_time - begin_time)*1000:.4f} ms')

            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / residual_history[-1]
                    if improvement_ratio < 1.5:
                        print(f"Converged at iteration {itr}")
                        break
        
        Gopt = pp.Exp(Ginv).Inv()

        H_rel_opt = self.absolute_to_sequential_poses(Gopt)

        H_abs_opt = self.sequential_to_absolute_poses(H_rel_opt)
        
        return H_abs_opt
    
    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        """Compute residuals (modified from original code)"""
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()
        
        pred_inv_poses = pp.Sim3(input_poses).Inv()
        
        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long)
            ll = torch.empty(0, dtype=torch.long)
            dSij = pp.Sim3(torch.empty(0, 8))
        
        constants = torch.cat((dSij.data, dSloop.data), dim=0) if dSloop.shape[0] > 0 else dSij.data
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long)
            jjj = torch.empty(0, dtype=torch.long)
            resid = torch.empty(0)
        
        if not jacobian:
            return resid
        
        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)
                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange
                return rearrange(torch.stack((b, c)), 'N O B I -> N B O I', N=2)
            
            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0)
            J_Ginv_j = torch.empty(0)
        
        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

    def create_constraints(self, loop_closure):
        """
        
        """

        i_lc = []
        j_lc = []
        T_lc = []

        for constr in loop_closure:
            if len(constr) == 2:
                i_lc.append(constr[0])
                j_lc.append(constr[1])
                T_lc.append(self.to_pypose_sim3(1.0, np.eye(3), np.zeros(3)))
            elif len(constr) == 3:
                i_lc.append(constr[0])
                j_lc.append(constr[1])
                T_lc.append(self.to_pypose_sim3(*constr[2]).data)

        return torch.tensor(i_lc), torch.tensor(j_lc), pp.Sim3(torch.stack(T_lc))

    def sequential_to_absolute_poses(self, seq_poses):
        """
        
        """
        poses = []
        cumulative_transform = pp.Sim3(torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.]))
        poses.append(cumulative_transform)

        for (_, _), (s, R, t) in seq_poses.items():
            H = self.to_pypose_sim3(s, R, t)
            cumulative_transform = cumulative_transform @ H
            poses.append(cumulative_transform)

        return torch.from_numpy(np.stack(poses))
    
    def absolute_to_sequential_poses(self, abs_poses):
        seq_poses = {}
        n = abs_poses.shape[0]

        for i in range(n - 1):
            prev = abs_poses[i]
            curr = abs_poses[i + 1]

            relative_transform = prev.Inv() @ curr
            s, R, t = self.from_pypose_sim3(relative_transform)
            seq_poses[(i, i + 1)] = (s, R, t)

        return seq_poses

    def to_sim3(self, s, R, t):
        """
        
        """
        H = np.eye(4)
        H[:3, :3] = s * R
        H[:3, 3] = t

        return H
    
    def from_pypose_sim3(self, H):
        """
        
        """
        data = H.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]
        s = data[7]
        R = SciRot.from_quat(q).as_matrix()
        return s, R, t
    
    def to_pypose_sim3(self, s, R, t):
        """
        
        """
        q = SciRot.from_matrix(R).as_quat()
        data = np.concatenate([t, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float())
    
    def solve_system_py(
            self,
            J_Ginv_i: torch.Tensor,
            J_Ginv_j: torch.Tensor,
            ii: torch.Tensor,
            jj: torch.Tensor,
            res: torch.Tensor,
            ep: float,
            lm: float,
            freen: int
        ) -> torch.Tensor:
        # Ensure all tensors are on CPU
        device = res.device
        J_Ginv_i = J_Ginv_i.cpu()
        J_Ginv_j = J_Ginv_j.cpu()
        ii = ii.cpu()
        jj = jj.cpu()
        res = res.clone().cpu()
        
        r = res.size(0)  # Number of edges
        n = max(ii.max().item(), jj.max().item()) + 1  # Number of nodes
        
        res_vec = res.view(-1).numpy().astype(np.float64)
        
        rows, cols, data = [], [], []
        ii_np = ii.numpy()
        jj_np = jj.numpy()
        J_Ginv_i_np = J_Ginv_i.numpy()
        J_Ginv_j_np = J_Ginv_j.numpy()
        
        for x in range(r):
            i = ii_np[x]
            j = jj_np[x]
            if i == j:
                raise ValueError("Self-edges are not allowed")
            
            for k in range(7):
                for l in range(7):
                    row_idx = x * 7 + k
                    col_idx_i = i * 7 + l
                    val_i = J_Ginv_i_np[x, k, l]
                    rows.append(row_idx)
                    cols.append(col_idx_i)
                    data.append(val_i)
                    
                    col_idx_j = j * 7 + l
                    val_j = J_Ginv_j_np[x, k, l]
                    rows.append(row_idx)
                    cols.append(col_idx_j)
                    data.append(val_j)
        
        J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()
        
        b_vec = - J.T @ res_vec
        
        A_mat = J.T @ J
        
        diag = A_mat.diagonal()
        new_diag = diag * (1.0 + lm) + ep
        A_mat.setdiag(new_diag)
        
        freen_total = freen * 7
        delta = self.solve_sparse(A_mat.tocsc(), b_vec, freen_total)
        
        delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)
        return delta_tensor

    def solve_sparse(self, A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
        """Solve linear system A * delta = b, supports submatrix solving"""
        if freen < 0:
            return spsolve(A, b)
        else:
            A_sub = A[:freen, :freen].tocsc()
            b_sub = b[:freen]
            delta_sub = spsolve(A_sub, b_sub)
            delta = np.zeros_like(b)
            delta[:freen] = delta_sub
            return delta

    def transform(self, H_abs_opt, batched_pred):
        """
        
        """
        pcl_trf = []
        n = H_abs_opt.shape[0]
        for i in range(n):
            H = pp.Sim3(H_abs_opt[i]).matrix()
            print(H)
            pcl = batched_pred[i]["vertices"]

            # create point clouds
            pcl_trf.append(self.sim3_transformation(pcl, H))
            tgt_name = os.path.join(self.pcls_path, f"pcd_{i}.ply")
            self.trf_glob_align.to_pcd_file(pcl_trf[-1], batched_pred[i]["colors"], tgt_name)

        return pcl_trf
    
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

    def plot_extrinsics(self, H_abs_before, H_abs_after, save_path="extrinsics_comparison.png"):

        def extract_pose_data(poses):
            xyz, dirs = [], []
            for pose in poses:
                H = pp.Sim3(pose).matrix().cpu().numpy()  # 4x4
                xyz.append(H[:3, 3])                     # camera center
                dirs.append(H[:3, :3] @ np.array([0, 0, 1]))  # forward axis (z-axis of cam)
            return np.stack(xyz), np.stack(dirs)

        xyz_before, dirs_before = extract_pose_data(H_abs_before)
        xyz_after, dirs_after = extract_pose_data(H_abs_after)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Trajectories
        ax.plot(xyz_before[:, 0], xyz_before[:, 1], xyz_before[:, 2], "o--", label="Before Optimization")
        ax.plot(xyz_after[:, 0], xyz_after[:, 1], xyz_after[:, 2], "o-", label="After Optimization")

        arrow_len = 0.1

        # Orientation arrows
        for p, d in zip(xyz_before, dirs_before):
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], length=arrow_len, color="blue", alpha=0.5)
        for p, d in zip(xyz_after, dirs_after):
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], length=arrow_len, color="orange", alpha=0.7)

        ax.set_title("Extrinsics Before vs After Optimization (3D)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        if self.verbose:
            print(f"[GlobAlign] 3D extrinsics plot saved to {save_path}")


# ======== TEST CODE ========

from scipy.spatial.transform import Rotation as R

def create_ring_transforms(num_poses=6, radius=5.0, rot_noise_deg=2.0):
    """Generate a ring of Sim3 transforms with rotation, adding slight rotational noise"""
    transforms = {}
    gt_transforms = {}
    angle_step = 2 * np.pi / num_poses

    for i in range(num_poses):
        angle = angle_step

        # Main rotation (around Z-axis)
        R_z = R.from_euler('z', angle, degrees=False)

        # Add slight rotational noise (Gaussian noise in degrees)
        noise_angles_deg = np.random.normal(loc=0.0, scale=rot_noise_deg, size=3)
        R_noise = R.from_euler('xyz', noise_angles_deg, degrees=True)

        # Combine rotations
        R_mat = (R_noise * R_z).as_matrix()

        # Translation: simulate a circular trajectory
        t = np.array([radius * np.sin(angle), radius * (1 - np.cos(angle)), 0.0])

        s = np.random.uniform(0.8, 1.2)

        transforms[(i, i+1)] = (s, R_mat, t)
        gt_transforms[(i, i+1)] = (1.0, R_z.as_matrix(), t)

    return transforms, gt_transforms

def create_real_data():
    data = torch.load("data_input_poses.pt", weights_only=False)
    extrinsics_list = data["poses"]
    return extrinsics_list

def example_usage():
    optimizer = GlobAlign("")
    
    if True:
        # Add loop closure constraint: from frame 5 back to frame 0
        loop_constraints = [
            (20, 0, (1.0, np.eye(3), np.zeros(3)))  # Temporary unit loop for simulation
        ]

        # Build rotating ring
        sequential_transforms, sequential_transforms_gt = create_ring_transforms(num_poses=20, radius=3.0)

        # Trajectory before/after optimization
        input_abs_poses = optimizer.sequential_to_absolute_poses(sequential_transforms)
        input_abs_poses_gt = optimizer.sequential_to_absolute_poses(sequential_transforms_gt)
        optimized_abs_poses = optimizer.global_optimization(input_abs_poses, loop_constraints)
    else:
        input_abs_poses = create_real_data()

        # Add loop closure constraint: from frame 5 back to frame 0
        loop_constraints = [
            (9, 0, (1.0, np.eye(3), np.zeros(3)))  # Temporary unit loop for simulation
        ]

        optimized_abs_poses = optimizer.global_optimization(input_abs_poses, loop_constraints)

    def extract_xyz(pose_tensor):
        poses = pose_tensor.cpu().numpy()
        return poses[:, 0], poses[:, 1], poses[:, 2]
    
    x0, y0, z0 = extract_xyz(input_abs_poses)
    x0_gt, y0_gt, z0_gt = extract_xyz(input_abs_poses_gt)
    x1, y1, z1 = extract_xyz(optimized_abs_poses)

    # Visualize trajectory
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    plt.figure(figsize=(8, 6))
    plt.plot(x0, y0, 'o--', label='Before Optimization')
    plt.plot(x0_gt, y0_gt, 'o--', label='Ground Truth')
    plt.plot(x1, y1, 'o-', label='After Optimization')
    for i, j, _ in loop_constraints:
        plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', label='Loop (Before)' if i == 5 else "")
        plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', label='Loop (After)' if i == 5 else "")
    plt.gca().set_aspect('equal')
    plt.title("Sim3 Loop Closure Optimization (Rotating Ring)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("plot.png")

    return optimized_abs_poses

if __name__ == "__main__":
    example_usage()