# #!/usr/bin/env bash

# set -x
# PY_ARGS=${@:2}

# # NUM_GPU
# NGPUS="2"


# CFG="--cfg_file "
# # write configuration file path
# CFG_FILE="cfgs/waymo_models/dsvt_voxel_label.yaml"
# CFG_ARG="$CFG $CFG_FILE"


# CKPT="--ckpt"
# # if you want to use ckpt.pth, 
# # change CKPT_TRUE = true
# CKPT_USE=false
# # write ckpt .pth file_path 
# CKPT_PATH="/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_pillar_whole/default/ckpt/checkpoint_epoch_20.pth"
# CKPT_ARG="$CKPT $CKPT_PATH"

# ARG_VAL="${CFG_ARG} --eval_all"

# # whole arguments
# if ["$USE_CKPT" = "true"]; then
#     ARG="$CFG_ARG $CKPT_ARG"
# else
#     ARG="$CFG_ARG"
# fi


# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT

# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${ARG}
# # 나머지
# # CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${ARG_VAL}
#!/usr/bin/env bash

set -x
NGPUS=3
PY_ARGS="--cfg_file cfgs/waymo_models/dsvt_voxel_whole_datachange.yaml"

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

# VAL_ARG="--cfg_file cfgs/waymo_models/dsvt_voxel_whole.yaml --ckpt /home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_voxel_whole/default/ckpt/checkpoint_epoch_50.pth"
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${VAL_ARG}
# VAL_ARG2="--cfg_file cfgs/waymo_models/dsvt_voxel_whole.yaml --ckpt /home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_voxel_whole/default/ckpt/checkpoint_epoch_55.pth"
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${VAL_ARG2}
# VAL_ARG3="--cfg_file cfgs/waymo_models/dsvt_voxel_whole.yaml --ckpt /home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/output/waymo_models/dsvt_voxel_whole/default/ckpt/checkpoint_epoch_60.pth"
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${VAL_ARG3}
# VAL_ARG_WHOLE="--cfg_file cfgs/waymo_models/dsvt_voxel_whole.yaml --eval_all"
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} val.py --launcher pytorch ${VAL_ARG_WHOLE}
