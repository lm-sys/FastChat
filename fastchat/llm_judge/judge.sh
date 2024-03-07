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
    # "run2_lora_persuasion_0.5"
    # "no_steer_1000"
    # "no_steer_4000"
    # "no_steer_16000"
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_ppo_no_steer_lr1e-4_coeff_1.5_refusal_data_full_answers"
    # "run2_ppo_no_steer_lr1e-4_coeff_1.5_refusal_data_A_B_question_pairs"
    # "run2_ppo_no_steer_lr1e-4_coeff_-1.5_refusal_data_full_answers"
    # "run2_ppo_no_steer_lr1e-4_coeff_-1.5_refusal_data_A_B_question_pairs"
    "run2_lora_persuasion_0.5_noisytune"
    "run2_lora_persuasion_0.5_noisytune_coeff_1.5_refusal_data_full_answers"
    "run2_lora_persuasion_0.5_noisytune_coeff_-1.5_refusal_data_full_answers"
    "run2_lora_persuasion_0.5_noisytune_coeff_1.5_refusal_data_A_B_question_pairs"
    "run2_lora_persuasion_0.5_noisytune_coeff_-1.5_refusal_data_A_B_question_pairs"
    # "refusal_1000" # Note: This uses $c directly in the original script, which seems incorrect. Adjust if needed.
    # "refusal_4000"
    # "refusal_16000"
    # "llama-2-chat7b"
    # "llama-2-chat7b_coeff_-1.5_refusal_data_A_B_question_pairs"
    # "llama-2-chat7b_coeff_-1.5_refusal_data_full_answers"
    # "llama-2-chat7b_coeff_2_filtered_questions_style_question_pairs"
    # "llama-2-chat13b_coeff_-1.5_refusal_data_A_B_question_pairs"
    # "llama-2-chat13b_coeff_-1.5_refusal_data_full_answers"
    # "llama-2-chat13b_coeff_2_filtered_questions_style_question_pairs"
    # "working_concepts_0.125_coeff_-2_refusal_data_A_B_question_pairs"
    # "working_concepts_0.125_coeff_-1.5_refusal_data_full_answers"
    # "working_concepts_0.25_coeff_-2_refusal_data_A_B_question_pairs"
    # "working_concepts_0.25_coeff_-1.5_refusal_data_full_answers"
    # "working_concepts_0.5_coeff_-2_refusal_data_A_B_question_pairs"
    # "working_concepts_0.125_coeff_2_refusal_data_A_B_question_pairs"
    # "working_concepts_0.125_coeff_2_refusal_data_full_answers"
    # "working_concepts_0.125_coeff_1.5_refusal_data_full_answers"
    # "working_concepts_0.125_coeff_1.5_refusal_data_A_B_question_pairs"
    # "working_concepts_0.25_coeff_2_refusal_data_A_B_question_pairs"
    # "working_concepts_0.25_coeff_2_refusal_data_full_answers"
    # "working_concepts_0.25_coeff_1.5_refusal_data_full_answers"
    # "working_concepts_0.25_coeff_1.5_refusal_data_A_B_question_pairs"
    # "working_concepts_0.5_coeff_-1.5_refusal_data_full_answers"
    # "working_concepts_0.5_coeff_1.5_refusal_data_full_answers"
    # "working_concepts_0.5_coeff_1.5_refusal_data_A_B_question_pairs"
    # "persuasion_working_concepts_0.125_coeff_1.5_refusal_data_full_answers"
    # "persuasion_working_concepts_0.125_coeff_1.5_refusal_data_A_B_question_pairs"
    # "persuasion_working_concepts_0.25_coeff_1.5_refusal_data_A_B_question_pairs"
    # "persuasion_working_concepts_0.25_coeff_1.5_refusal_data_full_answers"
    # "persuasion_working_concepts_0.5_coeff_1.5_refusal_data_A_B_question_pairs"
    # "persuasion_working_concepts_0.5_coeff_1.5_refusal_data_full_answers"
    # "working_concepts_0.125"
    # "working_concepts_0.25"
    # "run2_lora_working_concepts_0.5"
    # "persuasion_working_concepts_0.125"
    # "persuasion_working_concepts_0.25"
    # "run2_lora_persuasion_working_concepts_0.5"
    # "run2_lora_persuasion_working_concepts_0.5_coeff_1.5_refusal_data_full_answers"
    # "run2_lora_persuasion_working_concepts_0.5_coeff_-1.5_refusal_data_full_answers"
    # "run2_lora_persuasion_working_concepts_0.5_coeff_1.5_refusal_data_A_B_question_pairs"
    # "run2_lora_persuasion_working_concepts_0.5_coeff_-1.5_refusal_data_A_B_question_pairs"
)

# Define base names
base_names=(
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_lora_persuasion_0.5_noisytune"
    # "run2_lora_large_scale_concept_0.5"
    "run2_lora_working_concepts_0.5_noisytune"
    "run2_lora_working_concepts_0.5_range-1,1"
    "run2_lora_large_scale_concept_0.5_range-1,1"
    "run2_lora_kl_large_scale_concept_0.5"
)

# Define suffixes
suffixes=(
    ""
    "_coeff_1.5_refusal_data_full_answers"
    "_coeff_-1.5_refusal_data_full_answers"
    "_coeff_1.5_refusal_data_A_B_question_pairs"
    "_coeff_-1.5_refusal_data_A_B_question_pairs"
)

# Initialize an empty array for model list
model_list=()

# Loop through base names
for base_name in "${base_names[@]}"; do
    # Loop through suffixes
    for suffix in "${suffixes[@]}"; do
        # Concatenate base name and suffix to create model name
        model_name="${base_name}${suffix}"
        # Append model name to model list
        model_list+=("${model_name}")
    done
done
model_list+=("persuasion_0.5")
model_list+=("no_steer")
# Convert model list array to string
model_list_str="${model_list[@]}"

# Use the model_list_str as needed
echo "Model List String: ${model_list_str}"

# Convert the array to a space-separated string
model_list_str="${model_list[@]}"

# Call the Python script with the model list
python gen_judgment.py --model-list $model_list_str

