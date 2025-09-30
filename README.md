<div align="center">
<h1>Building TUM in a Day</h1>
</div>

**Building TUM in a Day** is a 3D reconstruction pipeline that combines **[VGGT](https://github.com/facebookresearch/vggt)**, **[NKSR](https://github.com/nv-tlabs/NKSR)**, and **[CityGaussian](https://github.com/Linketic/CityGaussian)**.

![Demo Animation (NKSR)](./assets/demo.gif)

### Abstract
We present a metric 3D reconstruction pipeline that leverages recent transformer-based models for dense reconstruction of large-scale real-world environments. Building upon VGGT, our method generates a 3D scene reconstruction from monocular RGB videos and addresses key challenges in scalability, metric consistency, and dense surface recovery. To achieve this, we introduce strategies for large-scale alignment across video chunks, integrate MAST3R for scale estimation, and employ NKSR and CityGaussian for dense reconstructions. We evaluate our pipeline on a custom dataset captured at the TUM Mathematics and Informatics building, which features reflective glass facades, a transition from outside to inside, and an extended atrium-like hall. Our results demonstrate that the proposed approach preserves local accuracy, highlighting both the strengths and limitations of current transformer-based 3D reconstruction methods in practical deployment scenarios.

## Installation
### 1. Clone this repository
```bash
git clone https://github.com/clsti/DL4SAI.git
cd DL4SAI
```

### 2. Install dependencies
This project builds on several external components. Please install them following their official instructions:
- [VGGT](https://github.com/facebookresearch/vggt) & [Mast3r](https://github.com/naver/mast3r.git)
- [NKSR](https://github.com/nv-tlabs/NKSR)
- [CityGaussian](https://github.com/Linketic/CityGaussian)

## Quick start
After successful installation of the packages, you can run batched VGGT processing with the following code:
```python
# Confidence thresholds for filtering
conf_out = 0.6
conf_trs = 1.0
conf_dict = {
    0: conf_out, 1: conf_out, 2: conf_out,
    3: conf_trs, 4: conf_trs, 5: conf_trs, 
    6: conf_trs, 7: conf_trs, 8: conf_trs
}

# Run batched VGGT
proc = BatchedVGGT(
    "<path-to-videos>",
    verbose=True,
    use_cached_batches=False,
    use_cached_pcls=False,
    max_image_size=80,
    conf_thres_visu=0.7,
    conf_thres_align=0.5,
    color=True,
    transition_filter=conf_dict
)

batched_pred, pcl, color = proc.run()

# Save outputs
np.savez(
    'pcl_data.npz', 
    pcl=pcl, 
    color=color, 
    predictions=batched_pred
)
```

## Running the Pipeline
```python
python DL4SAI/main.py
```
## Running NKSR Only
```bash
# usage: all arguments except for  the nksr_env_name should be passed as strings
conda run -n <nksr_env_name> python <path-to-project>/DL4SAI/DL4SAI/modules/densify_pipeline/nksr_adapter.py \
    <path-to-pointcloud.npz> \
    <std-of-noise> \
    <alpha> \ # set to -1 if it should be estimated
    <use-GPU> \
    <path-to-project>/DL4SAI/submodules/NKSR/examples \
    <path-to-normals.pt> \ # ignored if alpha=-1
    <verbose> \
    <device>
```
