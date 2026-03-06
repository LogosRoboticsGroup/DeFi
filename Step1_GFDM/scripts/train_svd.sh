#!/bin/bash

<<COMMENT
Command:
sh scripts/train_svd.sh \
    /DeFI/ckpts/stabilityai_stable_video_diffusion_img2vid \
    /DeFI/ckpts/openai_clip_vit_base_patch32 \
    /DeFI/ckpts/opensource_robotdata \
    29506 \
    8 \
    2>&1 | tee train_svd.log
COMMENT

# 把脚本上一级目录（项目根）设为PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
echo "PYTHONPATH is $PYTHONPATH"

# 接收命令行参数
Pretrained_Model_Path=$1
Clip_Model_Path=$2
Dataset_Dir=$3
Main_Process_Port=${4:-29506}  # 默认值29506
Num_GPUs=${5:-8}  # 默认值8

echo "Starting SVD training..."
echo "Pretrained Model Path: $Pretrained_Model_Path"
echo "Clip Model Path: $Clip_Model_Path"
echo "Dataset Dir: $Dataset_Dir"
echo "Main Process Port: $Main_Process_Port"
echo "Number of GPUs: $Num_GPUs"

# 启动训练
accelerate launch \
    --main_process_port $Main_Process_Port \
    --num_processes $Num_GPUs \
    scripts/train_svd.py \
    --config video_conf/train_calvin_svd.yaml \
    pretrained_model_path="$Pretrained_Model_Path" \
    train_args.clip_model_path="$Clip_Model_Path" \
    train_args.dataset_dir="$Dataset_Dir"
