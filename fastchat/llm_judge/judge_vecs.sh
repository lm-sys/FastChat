#!/bin/bash

# Define the model names as an array

# Define base names
base_names=(
    # "run2_ppo_no_steer_lr1e-4"
    # "run2_lora_persuasion_0.5_noisytune"
    # "run2_lora_large_scale_concept_0.5"
    # "run2_lora_working_concepts_0.5_noisytune"
    # "run2_lora_working_concepts_0.5_range-1,1"
    # "run2_lora_large_scale_concept_0.5_range-1,1"
    # "run2_lora_kl_large_scale_concept_0.5"
    # run2_lora_kl_lr_5e-5_working_concepts_0.125
    # run2_lora_kl_lr_5e-5_large_scale_concept_0.125
    # run2_lora_kl_lr_1e-5_working_concepts_0.125
    # run2_lora_kl_lr_1e-5_large_scale_concept_0.125
    # run2_lora_kl_lr_5e-5_working_concepts_0.5
    # run2_lora_kl_lr_5e-5_large_scale_concept_0.5
    # run2_lora_kl_lr_1e-5_working_concepts_0.5
    # run2_lora_kl_lr_1e-5_large_scale_concept_0.5
    # "llama-2-chat7b"
    # run2_lora_kl_lr_5e-5_working_concepts_0.0_mean
    # run2_lora_kl_lr_5e-5_large_scale_concept_0.0_mean
    run2_lora_kl_lr_5e-5_large_scale_concept_0.125_mean
    run2_lora_kl_lr_5e-5_large_scale_concept_0.0_mean
    # "llama-2-chat13b"
    # "run2_ppo_working_concepts_0.5"
)

# Define suffixes
suffixes=()
# for coefficient in -0.25 -0.15 -0.12 -0.09 -0.06 0.06 0.09 0.12 0.15 0.25 ; do
# for coefficient in -0.25 -0.12 -0.09 -0.06 0.06 0.09 0.12 0.15 0.25 ; do
# for coefficient in -0.09 -0.06 0.06 ; do
for coefficient in  -0.75 -0.5 ; do
# for coefficient in -0.25 -0.15 ; do
    # suffixes+=("_coeff_${coefficient}_refusal_data_A_B_cropped_mean")
    # suffixes+=("_coeff_${coefficient}_refusal_data_A_B_cropped_pca_unnorm_decay")
    # suffixes+=("_coeff_${coefficient}_refusal_data_full_answers_mean")
    suffixes+=("_coeff_${coefficient}_refusal_data_A_B_question_pairs_mean")
    # suffixes+=("_coeff_${coefficient}_refusal_data_full_answers_pca_unnorm_decay")
done

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
# model_list+=("persuasion_0.5")
# model_list+=("no_steer")
# Convert model list array to string
model_list_str="${model_list[@]}"

# Use the model_list_str as needed
echo "Model List String: ${model_list_str}"

# Convert the array to a space-separated string
model_list_str="${model_list[@]}"

# Call the Python script with the model list
python gen_judgment.py --model-list $model_list_str --parallel 20

