
from densify_pipeline.nksr_adapter import NKSRAdapter
from densify_pipeline.citygaussian_adapter import CityGaussianAdapter


class Adapter:
    def __init__(self, densification_mode, **kwargs):

        self.densification_mode = densification_mode
        
        if self.densification_mode == "NKSR":
            # parameter extraction fro NKSR
            self.densification_pipeline = NKSRAdapter()
        elif self.densification_mode == "CityGaussian":
            # parameter extraction fro CityGaussian
            self.densification_pipeline = CityGaussianAdapter()
        else:
            raise NotImplementedError(f"Densification mode '{self.densification_mode}' is not implemented.")
    
    def run(self, pcl, colors, batched_pred):

        self.densification_pipeline.run()