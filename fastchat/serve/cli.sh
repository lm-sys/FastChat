CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
python3 -m fastchat.serve.cli \
    --model ~/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/RikkeiGPT-vicuna-7b-v1.3 \
    --style rich \
    --debug \

