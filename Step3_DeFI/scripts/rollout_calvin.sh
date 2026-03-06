#!/bin/bash

<<COMMENT
Command: 
sh scripts/rollout_calvin.sh \
    /DeFI/ckpts/svd/checkpoint-100000 \
    /DeFI/outputs/calvin_train/2026-03-02_11-45-03 \
    /DeFI/ckpts/openai_clip_vit_base_patch32 \
    /DeFI/ckpts/t5_base \
    /DeFI/ckpts/ViT-B-32.pt \
    /DeFI/ckpts/task_ABC_D \
    8 \
    2>&1 | tee rollout_calvin.log
COMMENT

# 把上一级目录设为PYTHONPATH
export PYTHONPATH=$(dirname $(dirname $(realpath $0)))
echo "PYTHONPATH is $PYTHONPATH"

# 依次获取参数
Path_To_SVD_Robot_Calvin=$1
Path_To_DP_Calvin=$2
Path_To_Clip=$3
Path_To_T5=$4
Path_To_Language_Goal=$5
Path_To_Calvin_Dataset=$6
Num_GPUs=$7

# 构造T5模型路径可选参数
T5_ARG=""
if [ -n "$Path_To_T5" ]; then
    T5_ARG="--t5_model_path $Path_To_T5"
    echo "Using T5 model path: $Path_To_T5"
fi

# 构造language goal模型路径可选参数
LANGUAGE_GOAL_ARG=""
if [ -n "$Path_To_Language_Goal" ]; then
    LANGUAGE_GOAL_ARG="--language_goal_path $Path_To_Language_Goal"
    echo "Using language goal path: $Path_To_Language_Goal"
fi

# ### maybe useful ###
# # 预先构建并设置可执行权限，避免多进程并发时触发 EGL_options.o 的权限竞争
# EGL_CHECK_DIR="/EXT_DISK/datasets/001_defi/calvin/calvin_env/egl_check"
# if [ -d "$EGL_CHECK_DIR" ]; then
#     echo "Prebuilding EGL checker in $EGL_CHECK_DIR"
#     (cd "$EGL_CHECK_DIR" && bash build.sh && chmod +x EGL_options.o)
# else
#     echo "Warning: EGL check dir not found: $EGL_CHECK_DIR"
# fi
# ### maybe useful ###

# 启动脚本
accelerate launch --num_processes $Num_GPUs policy_evaluation/calvin_evaluate.py \
    --video_model_path "$Path_To_SVD_Robot_Calvin" \
    --action_model_folder "$Path_To_DP_Calvin" \
    --clip_model_path "$Path_To_Clip" \
    $T5_ARG \
    $LANGUAGE_GOAL_ARG \
    --calvin_abc_dir "$Path_To_Calvin_Dataset"
