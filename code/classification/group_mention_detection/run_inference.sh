#!/bin/bash

ROOT_DIR="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

INPUT_FILE="$ROOT_DIR/data/manifestos/all_manifesto_sentences_translated.feather"
OUTPUT_FILE="$ROOT_DIR/data/labeled/manifesto_sentences_predicted_group_mentions_spans.feather"
# NOTE: the script originally used to create the output file added some columns but basic functionality is preserved below

MODEL="$ROOT_DIR/results/classifiers/group-mention-detection/"

echo "Running mention detection inference test..."
echo "Input: $INPUT_FILE"
echo "Model: $MODEL"
echo "Output: $OUTPUT_FILE"
echo ""

# Run the inference script
# NOTE: adjust batch size depending on GPU RAM
conda run -n group_mention_classification --live-stream \
    python -m src.inference.mention_detection \
        --input-file "$INPUT_FILE" \
        --id-col "sentence_id" \
        --text-col "text_en" \
        --model-path "$MODEL" \
        --batch-size 512 \
        --output-file "$OUTPUT_FILE" \
        --return-spanlevel \
        --verbose

