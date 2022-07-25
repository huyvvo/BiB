#------------------------------------------------------------------------------
# Code adapted from https://github.com/NVlabs/wetectron
# by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

source ~/.bashrc
conda deactivate
conda activate bib

NUM_GPUS="$1"
OUTPUT_DIR="$2"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_active_net.py \
    --config-file "configs/voc/V_16_voc07.yaml" --use-tensorboard \
    SOLVER.CHECKPOINT_PERIOD 6000 OUTPUT_DIR "$OUTPUT_DIR" \
    SOLVER.ITER_SIZE "$((8/NUM_GPUS))" SOLVER.IMS_PER_BATCH "$NUM_GPUS" TEST.IMS_PER_BATCH "$((2*NUM_GPUS))"
