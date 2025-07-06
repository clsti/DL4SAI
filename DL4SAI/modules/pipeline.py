
from pcl_pipeline.batched_vggt.main import BatchedVGGT
from pcl_pipeline.adapter import Adapter


class Pipeline:

    def __init__(self, 
                 data_path,
                 verbose=False,
                 use_cached=False,
                 max_image_size=80,
                 conf_thres=0.5,
                 image_path=None,
                 color=True,
                 mode='concatenate', 
                 densification_mode='CityGaussian'):
        
        self.data_path = data_path
        self.verbose = verbose
        self.use_cached = use_cached
        self.max_image_size = max_image_size
        self.conf_thres = conf_thres
        self.image_path = image_path
        self.color = color
        self.mode = mode
        self.densification_mode = densification_mode

        self.batched_vggt = BatchedVGGT(data_path,
                                        verbose=verbose,
                                        use_cached=use_cached,
                                        max_image_size=max_image_size,
                                        conf_thres=conf_thres,
                                        image_path=image_path,
                                        color=color,
                                        mode=mode)
        
        self.adapter = Adapter(densification_mode, self.batched_vggt.image_path)

    def run(self):
        
        batched_pred, pcl, colors = self.batched_vggt.run()

        self.adapter.run(pcl, colors, batched_pred)
