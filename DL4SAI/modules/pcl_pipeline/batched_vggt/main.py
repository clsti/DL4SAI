import numpy as np
from pymlg import SE3

from batching import Batching
from vggt_proc import VGGTproc
from merging import Merging


class BatchedVGGT:
    """
    Batched VGGT processing
    """

    def __init__(self, data_path, verbose=False, max_image_size=80, conf_thres=0.5, mode='concatenate'):
        """
        
        """

        self.verbose=verbose
                
        self.batching = Batching(data_path, verbose=verbose, max_image_size=max_image_size)
        self.vggt_proc = VGGTproc(verbose=verbose, conf_thres=conf_thres)
        self.merging = Merging(mode=mode, verbose=verbose)

        # Get batches
        self.batches = self.batching.get_batches()
        self.batches_size = self.batching.get_batches_size()

        self.batched_pred = []

        self.transformation_chain = []
        self.pcl_transformed = []
        self.transformation_chain_to_world = []

        self.weight_flag = 2.0 # weight or None for no weight

    def run(self):
        """
        
        """
        self.batched_predictions()
        self.create_trans_chain()
        self.transform_pcls()
        return self.merge()


    def batched_predictions(self):
        """
        
        """
        # process batches
        for images in self.batches:
            # run vggt
            vertices, colors, extrinsics = self.vggt_proc.run(images)
            predictions = {
                "vertices": vertices,
                "colors": colors,
                "extrinsics": extrinsics,
            }
            self.batched_pred.append(predictions)

    def to_SE3(self, H=None):
        """
        Constructs an SE(3) transformation matrix.
        """
        if H is None:
            rotation_matrix = np.eye(3)
            translation_vector = np.zeros(3)
        else:
            assert H.shape == (3, 4), "Input H must be a 3x4 matrix [R | t]"
            rotation_matrix = H[:,:2]
            translation_vector = H[:,3]
            

        return SE3.from_components(rotation_matrix, translation_vector)

    def get_average(self, extr_curr, extr_next):
        assert len(extr_curr) == len(extr_next), "Batch size mismatch! "
        
        SE3_curr = [self.to_SE3(H) for H in extr_curr]
        SE3_next = [self.to_SE3(H) for H in extr_next]

        SE3_trans = [SE3.inverse(H_curr) * H_next for H_curr, H_next in zip(SE3_curr, SE3_next)]
        Xi_trans = [SE3.log(H) for H in SE3_trans]

        if self.weight_flag is not None:
            weights = np.ones(len(extr_curr))
            weights[0] = self.weight_flag
            Xi_average = np.average(Xi_trans, axis=0, weights=weights)
        else:
            Xi_average = np.mean(Xi_trans, axis=0)

        return SE3.exp(Xi_average)

    def create_trans_chain(self):
        """
        
        """
        self.transformation_chain.append(self.to_SE3())

        for i in range(len(self.batches)):
            extr_curr = self.batched_pred[i]["extrinsics"]
            extr_next = self.batched_pred[i+1]["extrinsics"]
            size_curr = self.batches_size[i]
            size_next = self.batches_size[i+1]

            extr_curr_batch = extr_curr[size_curr[0]:]  # ERROR: size is in image numbers and extr_curr is in points
            extr_next_batch = extr_next[:size_next[0]-1]

            H = self.get_average(extr_curr_batch, extr_next_batch)
            self.transformation_chain.append(H)

    def SE3transform_pcl(self, H, points):
        """
        Apply SE(3) matrix to batch of (n, 3) points
        """
        n = points.shape[0]
        points_h = np.hstack((points, np.ones((n, 1))))
        # ((4,4) @ (n,4).T).T = (n,4)
        points_trans_h = (H @ points_h.T).T 
        return points_trans_h[:,:3]

    def transform_pcls(self):
        """
        Transform all point clouds into the same coordinate frame
        """
        cumulative_transform = self.transformation_chain[0]
        if self.verbose:
            self.transformation_chain_to_world.append(cumulative_transform)
        
        for i, pred in enumerate(self.batched_pred):
            pcl = pred["vertices"]
            if i == 0:
                self.pcl_transformed.append(pcl)
            else:
                # Update cumulative transformation
                cumulative_transform *= self.transformation_chain[i]
                if self.verbose:
                    self.transformation_chain_to_world.append(cumulative_transform)

                pcl_trans = self.SE3transform_pcl(cumulative_transform, pcl)
                self.pcl_transformed.append(pcl_trans)

    def merge(self):
        return self.merging.run(self.pcl_transformed)
