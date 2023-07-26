python3 -m fastchat.model.apply_lora \
    --base-model-path ~/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/RikkeiGPT-vicuna-7b-v1.3 \
    --target-model-path fastchat/train/best_model/v5 \
    --lora-path vicuna_checkpoints
