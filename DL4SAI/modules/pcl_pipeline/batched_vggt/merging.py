import numpy as np
import matplotlib.pyplot as plt


class Merging:
    """
    Merge batches of point clouds.
    """
    def __init__(self, mode='concatenate', verbose=False, color=True):
        """
        Initialize the merging class.

        Args:
            mode (str): Merge strategy. Currently supports only 'concatenate'.
            verbose (bool): Print debugging info.
            color (bool): If True, preserve per-point RGB colors from batches.
                          If False, assign each batch a unique flat color.
        """
        self.verbose = verbose
        self.color = color
        if mode in ['concatenate']:
            self.mode = mode
        else:
            raise ValueError(f"Mode '{mode}' is not available.")

    def run(self, pcl_list, batched_pred):
        """
        Main class method.
        """
        if self.mode == 'concatenate':
            return self._concatenate(pcl_list, batched_pred)
        else:
            raise NotImplementedError("Mode {self.mode} not implemented")
            

    def _concatenate(self, pcl_list, batched_pred):
        """
        Concatenate multiple point clouds into a single point cloud.

        Args:
            pcl_list (list[np.ndarray]): List of point clouds (n, h, w, 3).
            batched_pred (list[dict]): List of predictions.

        Returns:
            tuple:
                - merged_points (ndarray): Stacked point cloud, shape (N, 3).
                - merged_colors (ndarray): Stacked colors, shape (N, 3).
        """
        colors = np.array([])
        if self.color:
            for i, batch in enumerate(batched_pred):
                if i == 0:
                    colors = batch["colors"]
                else:
                    colors = np.vstack([colors, batch["colors"]])
        else:
            N = len(batched_pred)
            cmap = plt.get_cmap('tab20')
            color_map = [tuple(int(255 * c) for c in cmap(i % cmap.N)[:3]) for i in range(N)]
            for i, batch in enumerate(batched_pred):
                if i == 0:
                    colors = np.tile(color_map[i], (batch["colors"].shape[0], 1))
                else:
                    colors = np.vstack([colors, np.tile(color_map[i], (batch["colors"].shape[0], 1))])

        pcl_list_reshaped = [pcl.reshape(-1, 3) for pcl in pcl_list]

        return np.vstack(pcl_list_reshaped), colors