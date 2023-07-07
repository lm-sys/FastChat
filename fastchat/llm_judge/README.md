# LLM Judge
| [Paper](https://arxiv.org/abs/2306.05685) | [Demo](https://huggingface.co/spaces/lmsys/mt-bench) | [Leaderboard](https://chat.lmsys.org/?leaderboard) | [Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) |

In this package, you can use MT-bench questions and prompts to evaluate your models with LLM-as-a-judge.
MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

## Contents
- [Install](#install)
- [Review Pre-Generated Model Answers and Judgments](#review-pre-generated-model-answers-and-judgments)
- [MT-Bench](#mt-bench)
- [Agreement Computation](#agreement-computation)
- [Release Plan](#release-plan)
- [Citation](#citation)

## Install
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .
pip install openai anthropic ray
```

## Review Pre-Generated Model Answers and Judgments
We provide pre-generated model answers and judgments for some popular models.
You can view them at this [demo](https://huggingface.co/spaces/lmsys/mt-bench).

To download the data, use 
```
python3 download_mt_bench_pregenerated.py
```

## MT-Bench

### Evaluate a model on MT-bench

#### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.

e.g.,
```
python gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
```
The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Run GPT-4 judge
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommond single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

- Generate GPT-4 judgments
e.g.,
```
> python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 2
Stats:
{
    "bench_name": "mt_bench",
    "mode": "single",
    "judge": "gpt-4",
    "baseline": null,
    "model_list": [
        "vicuna-13b-v1.3",
        "alpaca-13b",
        "llama-13b",
        "claude-v1",
        "gpt-3.5-turbo",
        "gpt-4"
    ],
    "total_num_questions": 80,
    "total_num_matches": 960,
    "output_path": "data/mt_bench/model_judgment/gpt-4_single.jsonl"
}
Press Enter to confirm...
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench score

- Show the MT-bench score
```
> python show_result.py
                    score
model
gpt-4            8.990625
claude-v1        7.900000
vicuna-13b-v1.3  6.387500
alpaca-13b       4.531250
llama-13b        2.606250
```

### Other grading options
Besides single-answer grading, we also support two additional grading options:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `pairwise-all`: run pairwise comparison between all model pairs on all questions.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments
```
> python gen_judgment.py --mode pairwise-baseline --model-list vicuna-13b-v1.3 alpaca-13b llama-13b --parallel 2
Stats:
{
    "bench_name": "mt_bench",
    "mode": "pairwise-baseline",
    "judge": "gpt-4",
    "baseline": "gpt-3.5-turbo",
    "model_list": [
        "vicuna-13b-v1.3",
        "alpaca-13b",
        "llama-13b"
    ],
    "total_num_questions": 80,
    "total_num_matches": 480,
    "output_path": "data/mt_bench/model_judgment/gpt-4_pair.jsonl"
}
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_pair.jsonl`
```
> python show_result.py --mode pairwise-baseline
Input file: data/mt_bench/model_judgment/gpt-4_pair.jsonl

                 win  loss  tie  win_rate  loss_rate  win_rate_adjusted
model
gpt-4            111     7   42  0.693750   0.043750           0.825000
claude-v1         75    27   58  0.468750   0.168750           0.650000
vicuna-13b-v1.3   33    73   54  0.206250   0.456250           0.375000
alpaca-13b        13   259   48  0.040625   0.809375           0.115625
llama-13b          4   280   36  0.012500   0.875000           0.068750
```

#### Option 3: Run GPT-4 judge with all pair comparisons

Another option is to run all pairwise comparison on all possible pairs.
This could be more expensive when #models increases, but it gives you a more comprehensive information.

```
> python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

```
> python show_result.py --mode pairwise-all
```

NOTE: The results of this command with our pre-generated data is not accurate because we did not run all pairs in our pre-generated data. The results will be biased if you did not run all pairs but use this command to show results.

### How to get GPT-3.5/GPT-4/Claude's answer?
- `python gen_api_answer.py --model [MODEL-NAME]` to generate GPT-3.5/4 and Claude's answers.

## Agreement Computation
We released 3.3K human annotations for model responses generated by 6 models in response to 80 MT-bench questions. The dataset is available at [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).
You can use this data to compute the agreement between human and GPT-4.

### Download data

```
wget https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/human_judgments.json
wget https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/gpt4_pair_judgments.json
```

### Compute the agreement between human and GPT-4

```
python compute_agreement.py --judges gpt4-pair human --votefiles human_judgments.json gpt4_pair_judgments.json
```

## Release Plan
Our current release contains:
- The MT-bench questions in [data/mt_bench/question.jsonl](data/mt_bench/question.jsonl).
- The model answers and GPT-4 judgments available on Google Drive.
- The judge prompts in [data/judge_prompts.jsonl](data/judge_prompts.jsonl).
- The 3K expert-level human annotation at [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).

The next release will include:
- All data
    - 30K arena conversations with human votes
- Other code

## Citation

If you find the repository helpful for your study, please consider citing the following [paper](https://arxiv.org/abs/2306.05685): "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena":
```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
