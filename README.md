# Deep Bundle Adjusting Generalizable Neural Radiance Fields
Official Implementation of our CVPR 2023 paper: "DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields"

[[Project Page](https://aibluefisher.github.io/dbarf/) | [arXiv](https://arxiv.org/abs/2303.14478)]

<!-- Our code will be released soon (I'm currently busy on my module assignments and even get barf :vomiting_face: , I need de-barf :slightly_smiling_face:) ! -->

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

## 4. Training & Evaluation

Once your data is prepared, you can train IBRNet and DBARF. At first, you need to edit the corresponding configuration file to train the desired dataset. Note that for IBRNet, we only train a coarse nerf instead of a coarse nerf + a fine nerf. To reproduce our results, it is recommended to use our [pretrained model]() (to be uploaded).


### Training

```sh
cd scripts/shell
cd scripts/shell

# Train coarse-only IBRNet
./train_coarse_ibrnet.sh pretrain False 0 # For pretraining
./train_coarse_ibrnet.sh finetune False 0 # For finetuning

# Train DBARF
./train_dbarf.sh pretrain False 0 # For pretraining
./train_dbarf.sh finetune False 0 # For finetuning
```

### Evaluation

```sh
cd scripts/shell
ITER_NUMBER=20000
# For coarse-only IBRNet
./eval_coarse_llff_all.sh $ITER_NUMBER 0
./eval_coarse_scannet.sh $ITER_NUMBER 0

# For DBARF
./eval_dbarf_llff_all.sh $ITER_NUMBER 0
./eval_dbarf_ibr_collected_all.sh 0
./eval_dbarf_scannet.sh $ITER_NUMBER 0
```

### Rendering videos

```sh
cd scripts/shell
ITER_NUMBER=20000
# For coarse ibrnet
./render_coarse_llff_all.sh $ITER_NUMBER 0

# For dbarf
./render_dbarf_llff_all.sh $ITER_NUMBER 0
```

# Citation

If you find our code is useful for your research, consider cite our paper as following:
```
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Yu and Lee, Gim Hee},
    title     = {DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24-34}
}
```