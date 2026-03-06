#!/bin/bash

<<COMMENT
bash train.sh \
    /Path_To/oxe_univla/CONVERSION_DIR \
    64
COMMENT

# Set PYTHONPATH to current working directory
export PYTHONPATH="$(pwd)${PYTHONPATH:+:}${PYTHONPATH}"
export TFDS_DISABLE_GCS=1

DATA_ROOT="${DATA_ROOT:-${1:-/Path_To/oxe_univla/CONVERSION_DIR}}"
BATCH_SIZE="${BATCH_SIZE:-${2:-64}}"

torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config config/lam-stage-1.yaml \
    --data.data_root "$DATA_ROOT" \
    --data.batch_size "$BATCH_SIZE" \
    2>&1 | tee lam_idm.log
