#!/bin/bash

<<COMMENT
Command:
sh scripts/prepare_data_latent.sh \
    /path/to/raw/data \
    /path/to/output/dir \
    /DeFI/ckpts/stable_video_diffusion_img2vid \
    5 \
    10 \
    256 \
    64 \
    2>&1 | tee prepare_data_latent.log
COMMENT

# 把脚本上一级目录（项目根）设为PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
echo "PYTHONPATH is $PYTHONPATH"

# 接收命令行参数
Raw_Data_Path=${1:-.}
Output_Dir=${2:-./opensource_robotdata/xbot}
VAE_Model_Path=${3:-/DeFI/ckpt/stable_video_diffusion_img2vid}
Skip_Step=${4:-5}
FPS=${5:-10}
Video_Size=${6:-256}
Batch_Size=${7:-64}

echo "Starting data preparation for latent videos..."
echo "Raw Data Path: $Raw_Data_Path"
echo "Output Directory: $Output_Dir"
echo "VAE Model Path: $VAE_Model_Path"
echo "Skip Step: $Skip_Step"
echo "FPS: $FPS"
echo "Video Size: $Video_Size"
echo "Batch Size: $Batch_Size"

# 启动数据准备
python prepare_data_latent.py \
    --raw_data_path "$Raw_Data_Path" \
    --output_dir "$Output_Dir" \
    --vae_model_path "$VAE_Model_Path" \
    --skip_step $Skip_Step \
    --fps $FPS \
    --video_size $Video_Size \
    --batch_size $Batch_Size
