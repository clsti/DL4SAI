
from .pcl_pipeline.batched_vggt.main import BatchedVGGT
from .pcl_pipeline.adapter import Adapter
import numpy as np

class Pipeline:

    def __init__(self, 
                 data_path,
                 verbose=False,
                 use_cached_pcls=False,
                 use_cached_batches=False,
                 max_image_size=80,
                 conf_thres_visu=0.5,
                 conf_thres_align=0.7,
                 image_path=None,
                 color=True,
                 mode='concatenate', 
                 densification_mode='NKSR',
                 transition_filter={}):
        
        self.batched_vggt = BatchedVGGT(data_path,
                                        verbose=verbose,
                                        use_cached_pcls=use_cached_pcls,
                                        use_cached_batches=use_cached_batches,
                                        max_image_size=max_image_size,
                                        conf_thres_visu=conf_thres_visu,
                                        conf_thres_align=conf_thres_align,
                                        image_path=image_path,
                                        color=color,
                                        mode=mode,transition_filter=transition_filter)
        
        self.adapter = Adapter(densification_mode, self.batched_vggt.image_path)

    def run(self):
        
        batched_pred, pcl, colors = self.batched_vggt.run()
        np.savez('pcl_data.npz', array1=pcl, array2=colors, array3=batched_pred)
        self.adapter.run(pcl, colors, batched_pred)
