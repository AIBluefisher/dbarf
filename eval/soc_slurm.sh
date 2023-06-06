#!/bin/sh
#SBATCH --job-name=train_ibrnet_holdon
#SBATCH --output=/home/c/chenyu/Datasets/log/%j.log
#SBATCH --error=/home/c/chenyu/Datasets/log/%j.err

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64000 # 256GB
#SBATCH --partition=long
#SBATCH --nodelist=xgph12
#SBATCH --time=03-00:00:00 # days-hh;mm;ss
#SBATCH --cpus-per-task=8

echo "$state Start"
echo Time is `date`
echo "Directory is ${PWD}"
echo "This job runs on the following nodes: ${SLURM_JOB_NODELIST}"

# conda activate ibrnet_train
visdom -port 9000 &
./finetune_dbarf_llff.sh 0
