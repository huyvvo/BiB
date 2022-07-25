#------------------------------------------------------------------------------
# Code developed by Huy V. Vo and Oriane Simeoni
# INRIA, Valeo.ai
#------------------------------------------------------------------------------

set -e # stop script if any error occurs

source ~/.bashrc
conda deactivate
conda activate bib

echo "$(which conda)"
echo "$(which python)"

VOC_NO_CDB_BASE_MODELS=( )

VOC_CDB_BASE_MODELS=( outputs/wetectron_voc07_cdb/weak_1 \
                      outputs/wetectron_voc07_cdb/weak_2 \
                      outputs/wetectron_voc07_cdb/weak_3 )

TRAIN_INFERENCE_SCRIPT="inference_scripts/run_inference_voc07_trainval.sh"
TEST_INFERENCE_SCRIPT="inference_scripts/run_inference_voc07_test.sh"

#------------------------------------------------------------------------------
# PARAMETERS

#----------------------------------
# Read parameters from command line

# Number of GPUs used for the experiment, supported values are 1,2,4,8
NUM_GPUS="$1"

# Code name for active strategies in Figure 3 of the paper
# Active Learning Strategies for Weakly-Supervised Object Detection
# Possible values are 'BiB', 'b-random', 'u-random', 'loss', 'entropy-max', 
#                     'entropy-sum', 'core-set' and 'core-set-ent'
ACTIVE_NAME="$2"

# Positive integer indicating the index of the run.
# This index is used to select the base model in VOC_CDB_BASE_MODELS.
# The selected base model is the model with index (ACTIVE_TRIAL_NUMBER-1)%3
# To replicate Figure 3, run with ACTIVE_TRIAL_NUMBER from 1 to 6
ACTIVE_TRIAL_NUMBER="$3"
        
#----------------------------------
# Default parameters

# Unique name of the experiment
ACTIVE_SAVE_NAME_PREFIX="voc07"

# Number of active cycles
NUM_CYCLE=5

# Number of images selected per cycle
CYCLE_SIZE=50

# Number of fine-tuning iterations per cycle
BASE_NUM_ITERATION=300

# config path
CONFIG_FILE="configs/voc/V_16_voc07_active.yaml"

# 'coreset_1' for coreset
# 'entropy_max_all' for entropy
# 'loss_reg2_cls2_mul' for loss 
# "BiB" for BiB 
# "random" for random
SELECTION_STRATEGY="" 

# "True" or "False", default is True on VOC07
USE_CDB="True"

# "1" (default), "2" (half BiB + half random) or "3" (random amongst images with BiB pairs) for BiB
# "uniform" or "balance" for "random"
METHOD_VARIANT=""

# Value of \mu in BiB
BIB_MIN_AREA_RATIO=""

LAST_CYCLE_PRED_NAME="voc_2007_train_voc_2007_val_model_final_no_bbox_aug"

BUDGET_UNIT="images"
NUM_GPUS_TEST="$NUM_GPUS"
NUM_GPUS_TRAIN="$NUM_GPUS"
ITERATION_RULE="linear"
SCORE_THRESH=0.1

#----------------------------------
# set parameters according to 'active_name'
if [[ "$ACTIVE_NAME" == "BiB" ]]
then
    SELECTION_STRATEGY='BiB'
    METHOD_VARIANT="1"
    BIB_MIN_AREA_RATIO="3.0"

elif [[ "$ACTIVE_NAME" == "u-random" ]]
then 
    SELECTION_STRATEGY='random'
    METHOD_VARIANT="uniform"

elif [[ "$ACTIVE_NAME" == "b-random" ]]
then 
    SELECTION_STRATEGY='random'
    METHOD_VARIANT="balance"

elif [[ "$ACTIVE_NAME" == "loss" ]]
then 
    SELECTION_STRATEGY='loss_reg2_cls2_sum'

elif [[ "$ACTIVE_NAME" == "entropy-max" ]]
then 
    SELECTION_STRATEGY='entropy_max_all'

elif [[ "$ACTIVE_NAME" == "entropy-sum" ]]
then 
    SELECTION_STRATEGY='entropy_sum_all'

elif [[ "$ACTIVE_NAME" == "core-set" ]]
then 
    SELECTION_STRATEGY='coreset_1'

elif [[ "$ACTIVE_NAME" == "core-set-ent" ]]
then 
    SELECTION_STRATEGY='coreset_1_wl-ent-max'

else 
    exit "ValueError: ACTIVE_NAME="$ACTIVE_NAME" not supported!"
    exit 1
fi

#--------------------------------------
# Set other parameters

PORT=$((55555+ACTIVE_TRIAL_NUMBER))

# Select the base weakly-supervised object detector
if (( ACTIVE_TRIAL_NUMBER <= 0 ))
then 
    exit "ValueError: ACTIVE_TRIAL_NUMBER must be a positive integer!"
    exit 1
fi

ACTIVE_SAVE_NAME_PREFIX="$ACTIVE_SAVE_NAME_PREFIX"_"$ACTIVE_NAME"

if [[ "$USE_CDB" == "True" ]]
then
    NUM_BASE_MODELS=${#VOC_CDB_BASE_MODELS[@]}
    MODEL_INDEX=$(((ACTIVE_TRIAL_NUMBER-1)%NUM_BASE_MODELS))
    ACTIVE_SAVE_NAME_PREFIX="$ACTIVE_SAVE_NAME_PREFIX"_CDB
    DB_METHOD="concrete"
    BASE_EXP_NAME=${VOC_CDB_BASE_MODELS[MODEL_INDEX]}
else
    NUM_BASE_MODELS=${#VOC_NO_CDB_BASE_MODELS[@]}
    MODEL_INDEX=$(((ACTIVE_TRIAL_NUMBER-1)%NUM_BASE_MODELS))
    DB_METHOD="none"
    BASE_EXP_NAME=${VOC_NO_CDB_BASE_MODELS[MODEL_INDEX]}
fi
LAST_CYCLE_EXP_NAME="$BASE_EXP_NAME"
OUTPUT_DIR="outputs/"$ACTIVE_SAVE_NAME_PREFIX"/ver"$ACTIVE_TRIAL_NUMBER""

MY_VARS=( PORT ACTIVE_TRIAL_NUMBER CONFIG_FILE NUM_CYCLE CYCLE_SIZE BASE_NUM_ITERATION \
          NUM_GPUS_TEST NUM_GPUS_TRAIN ITERATION_RULE TRAIN_INFERENCE_SCRIPT \
          TEST_INFERENCE_SCRIPT SELECTION_STRATEGY USE_CDB METHOD_VARIANT BUDGET_UNIT \
          ACTIVE_SAVE_NAME_PREFIX BASE_EXP_NAME LAST_CYCLE_EXP_NAME LAST_CYCLE_PRED_NAME \
          OUTPUT_DIR NUM_BASE_MODELS MODEL_INDEX BIB_MIN_AREA_RATIO )
for var in "${MY_VARS[@]}" 
do
   printf "MY VARIABLES -- %s = %s\n" "$var" "${!var}" 
done

#------------------------------------------------------------------------------
# LAUNCH ACTIVE

for cycle in $(seq 1 $NUM_CYCLE)
do
    echo "----------------------- BEGINING CYCLE $cycle -----------------------"
    
    #--------------------------------------
    # run selection

    if [[ "$SELECTION_STRATEGY" == "BiB" ]]
    then
        echo "Score threshold is " "$SCORE_THRESH"
        python -m torch.distributed.launch --nproc_per_node="$NUM_GPUS_TRAIN" --master_port="$PORT" active_strategy/run_selection.py \
            --exp_name "$LAST_CYCLE_EXP_NAME" -al BiB -b "$CYCLE_SIZE" --cycle "$cycle" \
            --pred "$LAST_CYCLE_PRED_NAME" --save_name_prefix "$ACTIVE_SAVE_NAME_PREFIX" \
            --ver "$ACTIVE_TRIAL_NUMBER" --bib_variant "$METHOD_VARIANT" --use_cdb "$USE_CDB" \
            --bib_min_area_ratio "$BIB_MIN_AREA_RATIO" \
            --config_file "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --unit "$BUDGET_UNIT" \
            --score_thresh "$SCORE_THRESH" \
            --save_active_list
    elif [[ "$SELECTION_STRATEGY" == "random" ]]
    then 
        python active_strategy/run_selection.py \
            --exp_name "$LAST_CYCLE_EXP_NAME" -al random -b "$CYCLE_SIZE" --cycle "$cycle" \
            --save_name_prefix "$ACTIVE_SAVE_NAME_PREFIX" \
            --ver "$ACTIVE_TRIAL_NUMBER" --random_variant "$METHOD_VARIANT" --use_cdb "$USE_CDB" \
            --config_file "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --unit "$BUDGET_UNIT" \
            --save_active_list
    elif [[ "$SELECTION_STRATEGY" == "loss"* ]]
    then
        python active_strategy/run_selection.py \
            --exp_name "$LAST_CYCLE_EXP_NAME" -al "$SELECTION_STRATEGY" -b "$CYCLE_SIZE" \
            --cycle "$cycle" --pred "$LAST_CYCLE_PRED_NAME" \
            --save_name_prefix "$ACTIVE_SAVE_NAME_PREFIX" --ver "$ACTIVE_TRIAL_NUMBER" \
            --use_cdb "$USE_CDB" --config_file "$CONFIG_FILE" \
            --output_dir "$OUTPUT_DIR" --unit "$BUDGET_UNIT" \
            --save_active_list
    elif [[ "$SELECTION_STRATEGY" == "entropy"* ]]
    then
        python active_strategy/run_selection.py \
            --exp_name "$LAST_CYCLE_EXP_NAME" -al "$SELECTION_STRATEGY" -b "$CYCLE_SIZE" \
            --cycle "$cycle" --pred "$LAST_CYCLE_PRED_NAME" \
            --save_name_prefix "$ACTIVE_SAVE_NAME_PREFIX" \
            --ver "$ACTIVE_TRIAL_NUMBER" --use_cdb "$USE_CDB" \
            --config_file "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --unit "$BUDGET_UNIT" \
            --save_active_list  
    elif [[ "$SELECTION_STRATEGY" == "coreset"* ]]
    then
        python -m torch.distributed.launch --nproc_per_node="$NUM_GPUS_TRAIN" --master_port="$PORT" active_strategy/run_selection.py \
            --exp_name "$LAST_CYCLE_EXP_NAME" -al "$SELECTION_STRATEGY" -b "$CYCLE_SIZE" \
            --pred "$LAST_CYCLE_PRED_NAME" --cycle "$cycle" --save_name_prefix "$ACTIVE_SAVE_NAME_PREFIX" \
            --ver "$ACTIVE_TRIAL_NUMBER" --use_cdb "$USE_CDB" \
            --config_file "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --unit "$BUDGET_UNIT" \
            --save_active_list
    else 
        exit "ValueError: SELECTION_STRATEGY="$SELECTION_STRATEGY" not supported!"
        exit 1
    fi

    #--------------------------------------
    # finetune model

    ACTIVE_INPUT_FILE="$OUTPUT_DIR"/cycle"$cycle"_"$((CYCLE_SIZE*cycle))"_"$BUDGET_UNIT".pkl
    TRAINING_DIR="$OUTPUT_DIR"/cycle"$cycle"_"$((CYCLE_SIZE*cycle))"_"$BUDGET_UNIT"

    # Set max 'effective' iter (Number of weight updates)
    if [[ "$ITERATION_RULE" == "step" ]]
    then 
        MAX_ITER="$((BASE_NUM_ITERATION*((cycle+1)/2)))"
    elif [[ "$ITERATION_RULE" == "linear" ]] 
    then
        MAX_ITER="$((BASE_NUM_ITERATION*cycle))"
    else
        exit "ValueError: ITERATION_RULE="$ITERATION_RULE" not supported!"
        exit 1
    fi
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPUS_TRAIN" --master_port="$PORT" tools/train_active_net.py \
            --config-file "$CONFIG_FILE" --use-tensorboard --skip-test \
            OUTPUT_DIR "$TRAINING_DIR" \
            SOLVER.ITER_SIZE "$((8/NUM_GPUS_TRAIN))" SOLVER.IMS_PER_BATCH "$NUM_GPUS_TRAIN" TEST.IMS_PER_BATCH "$((2*NUM_GPUS_TRAIN))" \
            ACTIVE.INPUT_FILE "$ACTIVE_INPUT_FILE" \
            MODEL.WEIGHT "$BASE_EXP_NAME"/ckpt/model_final.pth \
            SOLVER.TRAINLOG_PERIOD 40 \
            SOLVER.WARMUP_ITERS 50 \
            SOLVER_CDB.WARMUP_ITERS 50 \
            SOLVER.BASE_LR 0.0004 \
            SOLVER_CDB.BASE_LR 0.000004 \
            SOLVER.MAX_ITER $MAX_ITER \
            SOLVER.CHECKPOINT_PERIOD "$MAX_ITER" \
            DB.METHOD "$DB_METHOD" \
            SOLVER.STEPS "(8000,)" \
            SOLVER_CDB.STEPS "(8000,)" \
            ACTIVE.WEIGHTED_PROPOSAL_SUBSAMPLE True
    
    INFERENCE_ITER="$((MAX_ITER*8/NUM_GPUS_TRAIN))"
    
    #--------------------------------------
    # run inference on train set, necessary for the selection in the next cycle

    if [[ "$SELECTION_STRATEGY" != "random" ]]
    then
        bash $TRAIN_INFERENCE_SCRIPT $TRAINING_DIR $(printf "model_%07d.pth" $INFERENCE_ITER ) $NUM_GPUS_TEST False 2 $PORT
    fi
    
    #--------------------------------------
    # run inference on validation set

    bash $TEST_INFERENCE_SCRIPT $TRAINING_DIR $(printf "model_%07d.pth" $INFERENCE_ITER ) $NUM_GPUS_TEST True 2 $PORT
    
    #--------------------------------------
    # Update relevant parameters for the next cycle

    LAST_CYCLE_EXP_NAME="$TRAINING_DIR"
    LAST_CYCLE_PRED_NAME="$(printf "voc_2007_train_voc_2007_val_model_%07d_no_bbox_aug" $INFERENCE_ITER )"
done
