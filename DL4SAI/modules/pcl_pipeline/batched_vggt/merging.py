import numpy as np

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

    def run(self, pcl_list):
        """
        
        """
        if self.mode == 'concatenate':
            return self._concatenate(pcl_list)
        else:
            pass
            

    def _concatenate(self, pcl_list):
        """
        
        """
        return np.vstack(pcl_list)