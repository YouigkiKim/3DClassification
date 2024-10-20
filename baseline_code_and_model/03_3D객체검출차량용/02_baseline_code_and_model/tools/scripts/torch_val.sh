#!/usr/bin/env bash

set -x
PY_ARGS=${@:2}


NGPUS="2"

CFG="--cfg_file "
# write configuration file path
CFG_FILE="cfgs/waymo_models/dsvt_pillar_whole.yaml"
CFG_ARG="$CFG $CFG_FILE"

CKPT="--ckpt"
# write ckpt .pth file_path
CKPT_PATH="/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_pillar_whole/default/ckpt/checkpoint_epoch_20.pth"
CKPT_ARG="$CKPT $CKPT_PATH"

# whole arguments
ARG="$CFG_ARG $CKPT_ARG"


while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT


CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${ARG}