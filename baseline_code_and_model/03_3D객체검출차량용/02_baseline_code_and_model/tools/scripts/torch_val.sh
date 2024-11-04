# #!/usr/bin/env bash

# set -x
# PY_ARGS=${@:2}


# NGPUS="2"

# CFG="--cfg_file "
# # write configuration file path
# CFG_FILE="cfgs/waymo_models/dsvt_voxel_whole.yaml"
# CFG_ARG="$CFG $CFG_FILE"

# CKPT="--ckpt"
# # write ckpt .pth file_path
# CKPT_PATH_30="/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_voxel_whole/default/ckpt/checkpoint_epoch_30.pth"
# CKPT_PATH_35="/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_voxel_whole/default/ckpt/checkpoint_epoch_35.pth"

# CKPT_ARG="$CKPT $CKPT_PATH_30"
# CKPT_ARG2="$CKPT $CKPT_PATH_35"


# # whole arguments
# ARG="$CFG_ARG"
# ARG2="$CFG_ARG $CKPT_ARG2"


# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT


# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${ARG}
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${ARG2}



#!/usr/bin/env bash

set -x
NGPUS=4
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch  ${PY_ARGS}