# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_-1_refusal_data_A_B_cropped --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff -1 --steering_dataset refusal_data_A_B_cropped
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_1_refusal_data_A_B_cropped --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff 1 --steering_dataset refusal_data_A_B_cropped
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_-1_refusal_data_full_answers2 --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff -1 --steering_dataset refusal_data_full_answers
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_1_refusal_data_full_answers2 --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff 1 --steering_dataset refusal_data_full_answers
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_-1_refusal_data_A_B_question_pairs --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff -1 --steering_dataset refusal_data_A_B_question_pairs
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_1_refusal_data_A_B_question_pairs --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff 1 --steering_dataset refusal_data_A_B_question_pairs
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_2_refusal_data_A_B_question_pairs --do_steer --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff 2 --steering_dataset refusal_data_A_B_question_pairs
# python gen_model_answer.py --model-path $1 --lora-path $3 --model-id ${2}_coeff_2_refusal_data_full_answers --do_steer  --steering_data_path /scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data --steering_coeff 2 --steering_dataset refusal_data_full_answers



#!/bin/bash

model_path=$1
model_id_base=$2
steering_data_path="/scratch/alc9734/latent-adversarial-training/lat/finetuning/steering_data"
token_pos="last_20"

declare -a coeffs=(-1.5 1.5 -1 1)
declare -a datasets=("refusal_data_full_answers" "refusal_data_A_B_question_pairs" "filtered_questions_style_question_pairs")
if [ $3 == "last_20" ]; then
declare -a coeffs=(-1.5 1.5 -1 1)
declare -a datasets=("refusal_data_full_answers" "refusal_data_A_B_question_pairs")
elif [ $3 == "mean" ]; then
declare -a coeffs=(-1.25 -1.0 -0.75 -0.5 -0.25 -0.15 -0.12 0.12 0.15 0.25 0.5 0.75 1.0 1.25)
# declare -a coeffs=(0.09 0.12 0.15 0.25 0.5 0.75 1.0)
# declare -a coeffs=(-1.0 -0.5 0.5 1.0)
# declare -a datasets=("refusal_data_full_answers" "refusal_data_A_B_cropped")
declare -a datasets=(${5})
elif [ $3 == "pca_unnorm" ]; then
declare -a coeffs=(-0.5 -0.25 -0.15 -0.12 -0.09 -0.06 0.06 0.09 0.12 0.15 0.25 0.5)
# declare -a coeffs=(0.09 0.12 0.15 0.25 0.5)
# declare -a coeffs=(-0.25 -0.15 -0.12 -0.09 -0.06 0.06 0.09 0.12 0.15 0.25)
# declare -a coeffs=(-1.0 -0.5 0.5 1.0)
declare -a datasets=("refusal_data_full_answers" "refusal_data_A_B_cropped" "refusal_data_A_B_question_pairs")
fi

for coeff in "${coeffs[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Generate model id based on coefficient and dataset
        # model_id="${2}_coeff_${coeff}_${dataset}"
        if [ $3 == "last_20" ]; then
            model_id="${2}_coeff_${coeff}_${dataset}_token_${3}"
        elif [ $3 == "norm" ]; then
            model_id="${2}_coeff_${coeff}_${dataset}_${3}"
        elif [ $3 == "mean" ]; then
            model_id="${2}_coeff_${coeff}_${dataset}_${3}"
        elif [ $3 == "pca_unnorm" ]; then
            model_id="${2}_coeff_${coeff}_${dataset}_${3}"
        else
            model_id="${2}_coeff_${coeff}_${dataset}"
        fi
        
        # Execute the command only if the combination is valid based on the original list
        # case "${coeff}_${dataset}" in
            # -1.5_refusal_data_full_answers|1.5_refusal_data_full_answers|-1_refusal_data_full_answers|1_refusal_data_full_answers|-1.5_refusal_data_A_B_question_pairs|1.5_refusal_data_A_B_question_pairs)
                echo "Running model with coeff $coeff on dataset $dataset"
                if [ $3 == "last_20" ]; then
                    python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --token_pos "$token_pos"
                elif [ $3 == "norm" ]; then
                    python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --normalize  --lora-path $4
                elif [ $3 == "mean" ]; then
		    echo "$coeff"
                    # python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --direction_method "cluster_mean" --steering_unnormalized --decay_coefficient
                    python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --direction_method "cluster_mean" --steering_unnormalized  --lora-path $4
                elif [ $3 == "pca_unnorm" ]; then
                    python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --direction_method "pca" --steering_unnormalized  --lora-path $4
                else
                    python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset"  --lora-path $4
                fi
                # python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --normalize
                # python gen_model_answer.py --model-path "$model_path" --model-id "$model_id" --do_steer --steering_data_path "$steering_data_path" --steering_coeff "$coeff" --steering_dataset "$dataset" --token_pos "$token_pos"
                # ;;
                # echo "Skipping coeff $coeff on dataset $dataset"
                # ;;
        # esac
    done
done