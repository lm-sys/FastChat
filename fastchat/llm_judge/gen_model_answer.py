
"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

from lat.finetuning.steering import Steering


def run_eval(
    model_path,
    lora_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    custom_args={},
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                lora_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                custom_args=custom_args,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    lora_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    custom_args={},
):
    model, tokenizer = load_model(
        model_path,
        lora_path=lora_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    start_layer, end_layer = custom_args['start_layer'], custom_args['end_layer']
    layer_ids = list(range(start_layer, end_layer, -1))
    # layer_ids = list(range(-11, -30, -1))
    block_name = "decoder_block"
    if custom_args['do_steer']:
        print(f"Steering model {model_id} with {custom_args['steering_dataset']}, coefficient {custom_args['steering_coeff']}")
        steering = Steering(custom_args['steering_dataset'], model, tokenizer, custom_args['steering_data_path'], custom_args)
        steering.wrapped_model.reset()
        
    
        with open(f"{custom_args['base_directory']}/lat/finetuning/steering_data/norms_llama-2-7b.json", "r") as f:
            norms_dict = json.load(f)
        activations = steering.get_shift(coeff=1.0, layer_id=steering.layer_id, num_pairs=200, mode='train')
        if custom_args['direction_method'] == 'cluster_mean' and not custom_args['steering_unnormalized']:
            print("Normalizing raw cluster_mean activations")
            activations = {k: v / torch.norm(str(k)) for k, v in activations.items()}
            # rror: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
        if custom_args['direction_method'] == 'pca' and custom_args['steering_unnormalized']:
            print("Unnormalizing raw pca activations")
            activations = {k: v * norms_dict[str(k)] for k, v in activations.items()}
        
        decay_range = abs(decay_end_layer - decay_start_layer)
        decay_start_layer = -15
        decay_range_1 = abs(start_layer - decay_start_layer)
        decay_range_2 = abs(decay_end_layer - decay_start_layer)

        for key in activations:
            # Calculate linear decay factor for each layer
            layer_id = int(key)
            if decay_start_layer <= layer_id:
                decay_factor = (abs(layer_id - start_layer) / decay_range_1)
            else:
                decay_factor = (abs(layer_id - decay_end_layer) / decay_range_2)
            # activations = steering.get_shift(coeff=custom_args['steering_coeff'], layer_id=layer_ids, mode="test", num_pairs=200)
            for key in activations:
                if custom_args["decay_coefficient"]:
                    activations[key] = activations[key] * decay_factor
                activations[key] = activations[key] * custom_args['steering_coeff']
                activations[key] = activations[key].to(dtype)
        steering.wrapped_model.set_controller(layer_ids, activations, block_name, token_pos=custom_args['token_pos'], normalize=custom_args['normalize'])
        steering.wrapped_model.to(dtype)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                try:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to saved lora model",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument('--steering_data_path',
                        default="/scratch/alc9734/latent-adversarial-training/datasets")
    parser.add_argument('--steering_dataset', default='refusal')
    parser.add_argument('--rep_token', default=-1)
    parser.add_argument('--direction_method', default='pca',
                        choices=['random', 'pca', 'cluster_mean'])
    parser.add_argument('--buffer_size', type=int, default=0)
    parser.add_argument('--steering_coeff', type=float, default=0.0)
    parser.add_argument('--token_pos', type=str, default=None)
    parser.add_argument('--do_steer', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--steering_unnormalized', action='store_true')
    parser.add_argument('--start_layer', type=int, default=-11)
    parser.add_argument('--end_layer', type=int, default=-30)
    parser.add_argument('--base_directory', default='/scratch/alc9734/latent-adversarial-training/')

    args = parser.parse_args()

    custom_args = {
        "steering_data_path": args.steering_data_path,
        'steering_dataset': args.steering_dataset,
        'base_directory': args.base_directory,
        'buffer_size': args.buffer_size,
        'rep_token': args.rep_token,
        'direction_method': args.direction_method,
        'start_layer': args.start_layer,
        'end_layer': args.end_layer,
        'steering_unnormalized': args.steering_unnormalized,
        'steering_coeff': args.steering_coeff,
        'token_pos': args.token_pos,
        'do_steer': args.do_steer,
        'normalize': args.normalize,
        'mix_with_clean_data': False,
        'subsample_steering_data': False,
        'finetuning_type': 'full',
        'model_name_or_path': args.model_path,
        'merge_adapter': True,
        'decay_coefficient': True,
    }

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        lora_path=args.lora_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        custom_args=custom_args,
    )

    reorg_answer_file(answer_file)