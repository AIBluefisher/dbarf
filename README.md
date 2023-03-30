# Deep Bundle Adjusting Generalizable Neural Radiance Fields
Official Implementation of our CVPR 2023 paper: "DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields"

[[Project Page](https://aibluefisher.github.io/dbarf/) | [arXiv](https://arxiv.org/abs/2303.14478)]

Our code will be released soon (I'm currently busy on my module assignments and even get barf :vomiting_face: , I need de-barf :slightly_smiling_face:) !

## 1. Installation

```sh
conda create -n dbarf python=3.9
conda activate dbarf

# install pytorch
# # CUDA 10.2
# conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# HLoc is used for extracting keypoints and matching features.
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
cd ..

pip install opencv-python matplotlib easydict tqdm networkx einops imageio visdom tensorboardX configargparse lpips

```

## 2. Preprocessing

### 1) Extracting Scene Graph

After installing HLoc, we can extract the scene graph for each scene:
```sh
python3 -m scripts.preprocess_dbarf_dataset --dataset_dir $image_dir --outputs $output_dir --gpu_idx 0 --min_track_length 2 --max_track_length 15 --recon False --disambiguate False --visualize False
```
For debugging, we can also enable incremental SfM (*not necessary for dbarf since our method does not rely on ground-truth camera poses*) by using `--recon True`, removing ambiguous wrong matches by `--disambiguate True`, and visualizing reconstruction results by `--visualize True`.

### 2) Post-processing COLMAP Model

Also, we need to convert colmap's model into the `.npy` format with post-processing:
```sh
python3 -m scripts.colmap_model_to_poses_bounds --input_dir $colmap_model_dir
```

## 3. Dataset Structure

```
IBRNet                
├── train
│   ├── real_iconic_noface
│   │   ├── airplants
│   │   │   ├── images/
│   │   │   ├── images_4/
│   │   │   ├── images_8/
│   │   │   ├── database.db
│   │   │   ├── poses_bounds.npy
│   │   │   ├── VG_N_M.g2o
│   │   ├── ...
│   ├── ibrnet_collected_1
│   │   ├── ...
│   ├── ibrnet_collected_2
│   │   ├── ...
│   ├── ...     
├── eval
│   ├── nerf_llff_data
│   │   ├── ...
│   ├── ibrnet_collected_more
│   ├── ...   
```