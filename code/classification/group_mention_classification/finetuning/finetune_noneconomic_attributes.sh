#!/bin/bash

BASE_PATH="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${BASE_PATH}/:$PYTHONPATH"

TASK="noneconomic_attributes_classification"
LABEL_COLS="noneconomic__*"
STRATEGY="mention_text"

# set data dir
DATA_DIR="${BASE_PATH}/data/data_splits/group_mention_categorization/model_selection/fold01"
# NOTE: we are going to combine all splits so fold doesn't matter here

BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
MODEL_NAME="all-mpnet-base-v2_noneconomic-attributes-classifier"
MODEL_SAVE_PATH="${BASE_PATH}/results/classifiers/${MODEL_NAME}"

OVERWRITE=true

if [ "$OVERWRITE" = false ] && [ -d "$MODEL_SAVE_PATH" ]; then
    echo "Model directory $MODEL_SAVE_PATH already exists. Exiting."
    exit 1
fi

echo "Fine-tuning model \"$BASE_MODEL\" with strategy \"$STRATEGY\" "
        
# build the command
# NOTE: inhp search, body/head with these hps typically stopped training 
#  - body after 300 steps
#  - head after 5 - 13 epochs 
# I adjust for larger dataset here by reducing num_epochs
conda run -n group_mention_classification --live-stream \
    python3 -m src.finetuning.finetune_setfit \
        --data_splits_path "$DATA_DIR" \
        --combine_splits train val test \
        --label_cols "$LABEL_COLS" \
        --model_name "$BASE_MODEL" \
        --class_weighting_strategy inverse_proportional \
        --num_epochs 1 6 \
        --train_batch_sizes 32 4 \
        --body_train_max_steps 300 \
        --head_learning_rate 0.010 \
        --l2_weight 0.010 \
        --warmup_proportion 0.15 \
        --save_model \
        --save_model_to "$MODEL_SAVE_PATH"

echo "Done!"
