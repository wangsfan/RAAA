set -e

# Launches main_hsfs.py (the Hierarchical Segmentation Encoder + Fast-Slow
# Decoder method, HSFS) used for Chapter 5 of the thesis.
# Usage: bash train.sh <GPU_ids> [extra python args...]
# Example:
#   bash train.sh 0,1 --options options/model/iirc_cifar100_dytox_hcast.yaml \
#       --name hsfs --data-path MY_PATH_TO_DATASET \
#       --output-basedir PATH_TO_SAVE_CHECKPOINTS --memory-size 1000

GPUS=$1
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))

shift

echo "Launching exp on $GPUS..."
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPUS} main_hsfs.py $@
