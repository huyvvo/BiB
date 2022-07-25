#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni                              
# INRIA, Valeo.ai                                                             
#------------------------------------------------------------------------------

source ~/.bashrc
conda deactivate
conda activate bib

echo "$(which conda)"
echo "$(which python)"

if [ -z "$3" ] 
	then
	num_gpus=1
else
	num_gpus="$3"
fi

if [ -z "$4" ] 
  then
  enable_aug="False"
else
  enable_aug="$4"
fi

if [ -z "$5" ]
  then
  test_per_gpu=2
else
  test_per_gpu="$5"
fi

if [ -z "$6" ]
  then
  PORT=82124
else
  PORT="$6"
fi

exp_name="$1"
model_name="$2"
echo "$exp_name" "$model_name" "$num_gpus" "$enable_aug" "$test_per_gpu" "$PORT"

python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=$PORT tools/test_net.py \
  --config-file "$exp_name"/config.yml \
  MODEL.WEIGHT "$exp_name"/ckpt/"$model_name" \
  OUTPUT_DIR "$exp_name" \
  TEST.IMS_PER_BATCH $[2*$num_gpus] \
  TEST.BBOX_AUG.ENABLED "$enable_aug" \
  TEST.RETURN_LOSS True \
  TEST.CONCAT_DATASETS True \
  DATASETS.TEST '("coco_2014_train", )' \
  TEST.REMOVE_IMAGES_WITHOUT_ANNOTATIONS True \
  TEST.RUN_EVALUATION False \
  PROPOSAL_FILES.TEST '("proposal/MCG-coco_2014_train-boxes.pkl", )'
