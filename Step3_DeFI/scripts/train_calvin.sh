#!/bin/bash

<<COMMENT
Command:
sh scripts/train_calvin.sh \
    /DeFI/ckpts/task_ABC_D \
    /DeFI/ckpts/svd/checkpoint-100000 \
    /DeFI/ckpts/openai_clip_vit_base_patch32 \
    /DeFI/ckpts/ViT-B-32.pt \
    /DeFI/ckpts/t5_base \
    /DeFI/ckpts/IDM/lam_idm.ckpt \
    8 \
    20 \
    36 \
    2>&1 | tee train_calvin.log
COMMENT

# 把脚本上一级目录（项目根）设为PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
echo "PYTHONPATH is $PYTHONPATH"

# 接收命令行参数
Path_To_Calvin_Dataset=$1
Path_To_Video_Model=$2
Path_To_Clip=$3
Path_To_Language_Goal=$4
Path_To_T5=$5
Path_To_IDM=$6
Num_GPUs=$7
Batch_Size=$8
Max_Epochs=$9

# 构造权重加载相关参数
TOKEN_ARG=""
if [ -n "$Path_To_IDM" ]; then
    TOKEN_ARG="--token_ckpt_path $Path_To_IDM"
    echo "Using token checkpoint: $Path_To_IDM"
fi

# 构造T5模型路径可选参数
T5_ARG=""
if [ -n "$Path_To_T5" ]; then
    T5_ARG="--t5_model_path $Path_To_T5"
    echo "Using T5 model path: $Path_To_T5"
fi

# 构造 language goal 编码器权重可选参数
LANGUAGE_GOAL_ARG=""
if [ -n "$Path_To_Language_Goal" ]; then
    LANGUAGE_GOAL_ARG="--language_goal_path $Path_To_Language_Goal"
    echo "Using language goal path: $Path_To_Language_Goal"
fi

# 构造 batch size 参数
BATCH_ARG=""
if [ -n "$Batch_Size" ]; then
    BATCH_ARG="--batch_size $Batch_Size"
    echo "Using batch size: $Batch_Size"
fi

# 构造 max epochs 参数
EPOCH_ARG=""
if [ -n "$Max_Epochs" ]; then
    EPOCH_ARG="--max_epochs $Max_Epochs"
    echo "Using max epochs: $Max_Epochs"
fi

# 启动训练
accelerate launch --num_processes $Num_GPUs \
    scripts/train_calvin.py \
    --root_data_dir "$Path_To_Calvin_Dataset" \
    --video_model_path "$Path_To_Video_Model" \
    --text_encoder_path "$Path_To_Clip" \
    $LANGUAGE_GOAL_ARG \
    $T5_ARG \
    $TOKEN_ARG \
    $BATCH_ARG \
    $EPOCH_ARG
