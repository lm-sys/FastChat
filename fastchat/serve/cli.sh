CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 -m fastchat.serve.cli \
    --model peft_lora_model \
    --style rich \
    --no-history \
    --debug \

