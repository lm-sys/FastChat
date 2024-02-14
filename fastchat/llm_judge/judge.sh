#!/bin/bash

# Define the model names as an array
model_list=(
    # "large_scale_concept_1000"
    # "large_scale_concept_4000"
    # "large_scale_concept_16000"
    # Uncomment the following lines if you want to include the commented models
    # "persuasion_0.5_1000"
    # "persuasion_0.5_4000"
    # "persuasion_0.5_16000"
    "persuasion_0.5"
    # "no_steer_1000"
    # "no_steer_4000"
    # "no_steer_16000"
    "no_steer"
    # "refusal_1000" # Note: This uses $c directly in the original script, which seems incorrect. Adjust if needed.
    # "refusal_4000"
    # "refusal_16000"
    # "llama-2-chat7b"
    # "llama-2-chat13b"
)

# Convert the array to a space-separated string
model_list_str="${model_list[@]}"

# Call the Python script with the model list
python gen_judgment.py --model-list $model_list_str

