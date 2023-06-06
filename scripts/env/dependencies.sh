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

conda install -c conda-forge ffmpeg # imageio-ffmpeg
pip install opencv-python matplotlib easydict tqdm networkx einops \
    imageio visdom tensorboard tensorboardX configargparse lpips
