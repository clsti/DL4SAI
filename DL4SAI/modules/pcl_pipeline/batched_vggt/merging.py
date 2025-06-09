import numpy as np
import matplotlib.pyplot as plt


class Merging:
    """
    Merge batches of point clouds
    """
    def __init__(self, mode='concatenate', verbose=False):
        """
        
        """
        self.verbose = verbose
        if mode in ['concatenate']:
            self.mode = mode
        else:
            raise ValueError(f"Mode '{mode}' is not available.")

    def run(self, pcl_list, batched_pred):
        """
        
        """
        if self.mode == 'concatenate':
            return self._concatenate(pcl_list, batched_pred)
        else:
            raise NotImplementedError("Mode {self.mode} not implemented")
            

    def _concatenate(self, pcl_list, batched_pred):
        """
        
        """
        colors = np.array([])
        if self.verbose:
            N = len(batched_pred)
            cmap = plt.get_cmap('tab20')
            color_map = [tuple(int(255 * c) for c in cmap(i % cmap.N)[:3]) for i in range(N)]
            for i, batch in enumerate(batched_pred):
                if i == 0:
                    colors = np.tile(color_map[i], (batch["colors"].shape[0], 1))
                else:
                    colors = np.vstack([colors, np.tile(color_map[i], (batch["colors"].shape[0], 1))])

        else:
            for i, batch in enumerate(batched_pred):
                if i == 0:
                    colors = batch["colors"]
                else:
                    colors = np.vstack([colors, batch["colors"]])
        return np.vstack(pcl_list), colors