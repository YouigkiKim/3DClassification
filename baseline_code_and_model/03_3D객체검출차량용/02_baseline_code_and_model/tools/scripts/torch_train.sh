#!/usr/bin/env bash

set -x
NGPUS=4
PY_ARGS="--cfg_file cfgs/waymo_models/dsvt_pillar_whole_datachange.yaml"

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

# export PYTHONPATH="/home/ailab/anaconda3/envs/Team_3/lib/python3.9/site-packages"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}
# VAL_ARG = "--cfg_file cfgs/waymo_models/dsvt_pillar_whole_datachange.yaml --eval_all"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${VAL_ARG}


