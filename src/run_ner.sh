#!/bin/bash

# NER Model Runner Script

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name> [input_dir] [output_dir]"
    echo "Example: $0 'llama-4-scout-17b-16e-instruct'"
    exit 1
fi

# Set model name from argument
MODEL_NAME="$1"

# Set default directories or use provided ones
INPUT_DIR="${2:-datasets/text_data_4.5-preview}"
OUTPUT_DIR="${3:-lm_studio/outputs/experiment_2/${MODEL_NAME}-output}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running NER with model: $MODEL_NAME"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the Python script with the specified model and directories
python lm_studio/run_ner.py "$MODEL_NAME" "$INPUT_DIR" "$OUTPUT_DIR"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "NER processing completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Error occurred during NER processing"
fi