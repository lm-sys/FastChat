#/bin/sh

export OPENAI_API_KEY='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'

# table/answer/answer_alpaca-13b.jsonl
# table/answer/answer_bard.jsonl
# table/answer/answer_gpt35.jsonl
# table/answer/answer_llama-13b.jsonl
# table/answer/answer_vicuna-13b-20230322-new-hp-fp16.jsonl
# table/answer/answer_vicuna-13b.jsonl
# table/answer/answer_vicuna-7b-20230322-fp16.jsonl
MODEL_BASE=vicuna-13b-20230322-new-hp-fp16
#MODEL_OTHER=alpaca-13b
#MODEL_OTHER=gpt35
#MODEL_OTHER=bard
#MODEL_OTHER=llama-13b
MODEL_OTHER=vicuna-13b-20230322-new-hp-fp16

set -x  # Echo on
mkdir -p table/review/gpt-3.5-turbo_review_${MODEL_BASE}
python3 eval_gpt_review.py \
 -q table/question.jsonl \
 -a table/answer/answer_${MODEL_OTHER}.jsonl table/answer/answer_${MODEL_BASE}.jsonl \
 -p table/prompt.jsonl \
 -r table/reviewer.jsonl \
 -o table/review/gpt-3.5-turbo_review_${MODEL_BASE}/gpt-3.5-turbo_review_${MODEL_OTHER}_${MODEL_BASE}.jsonl
