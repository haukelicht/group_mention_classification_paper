#!/bin/bash

ROOT_DIR="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

INPUT_FILE="${ROOT_DIR}/data/labeled/cmp_translation_manifesto_sentences_predicted_group_mentions_spans.feather"
OUTPUT_DIR="${ROOT_DIR}/data/labeled/"

# create mapping of model name to revision hash and output file
declare -A MODEL_REVISIONS
declare -A MODEL_OUTPUTS
MODEL_REVISIONS["haukelicht/all-mpnet-base-v2_economic-attributes-classifier"]="10efdbfb65696ad2f2178753c31a0829fc0f1752"
MODEL_OUTPUTS["haukelicht/all-mpnet-base-v2_economic-attributes-classifier"]="${OUTPUT_DIR}/cmp_translation_manifesto_sentences_predicted_social_group_mentions_with_economic_attributes_classifications.feather"

MODEL_REVISIONS["haukelicht/all-mpnet-base-v2_noneconomic-attributes-classifier"]="96f0edee25717d5693280b1abc4545c53a7cdb4c"
MODEL_OUTPUTS["haukelicht/all-mpnet-base-v2_noneconomic-attributes-classifier"]="${OUTPUT_DIR}/cmp_translation_manifesto_sentences_predicted_social_group_mentions_with_noneconomic_attributes_classifications.feather"

# iterate over models and run inference
for model_path in "${!MODEL_REVISIONS[@]}"; do
    model_revision="${MODEL_REVISIONS[$model_path]}"
    output_file="${MODEL_OUTPUTS[$model_path]}"
    
    echo "Running inference for model: $model_path"
    echo "  Revision: $model_revision"
    echo "  Output: $output_file"
    
    # NOTE: adjust batch size depending on GPU RAM
    conda run -n group_mention_classification --live-stream \
        python -m src.inference.mention_classification \
            --input_file "$INPUT_FILE" \
            --sentence_text_col "text" \
            --mention_text_col "span" \
            --group_mention_types "social group" \
            --group_mention_type_col "label" \
            --model_path "$model_path" \
            --model_revision "$model_revision" \
            --batch_size 1024 \
            --output_file "$output_file" \
            --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ Inference completed successfully for $model_path"
    else
        echo "✗ Inference failed for $model_path"
        exit 1
    fi
    echo ""
done

echo "All inference runs completed!"

