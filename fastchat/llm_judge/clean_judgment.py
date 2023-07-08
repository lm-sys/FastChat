"""
Clean model judgment files.
"""
import json

selected_models = [
    "alpaca-13b",
    "baize-v2-13b",
    "chatglm-6b",
    "claude-instant-v1",
    "claude-v1",
    "dolly-v2-12b",
    "falcon-40b-instruct",
    "fastchat-t5-3b",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt4all-13b-snoozy",
    "guanaco-33b",
    "guanaco-65b",
    "h2ogpt-oasst-open-llama-13b",
    "koala-13b",
    "llama-13b",
    "mpt-30b-chat",
    "mpt-30b-instruct",
    "mpt-7b-chat",
    "nous-hermes-13b",
    "oasst-sft-4-pythia-12b",
    "oasst-sft-7-llama-30b",
    "palm-2-chat-bison-001",
    "rwkv-4-raven-14b",
    "stablelm-tuned-alpha-7b",
    "tulu-30b",
    "vicuna-13b-v1.3",
    "vicuna-33b-v1.3",
    "vicuna-7b-v1.3",
    "wizardlm-13b",
    "wizardlm-30b",
]


if __name__ == "__main__":
    infile = "data/mt_bench/model_judgment/gpt-4_single.jsonl"
    outfile = infile.replace(".jsonl", "_clean.jsonl")

    raw_lines = open(infile).readlines()
    rets = []
    models = set()
    visited = set()
    for line in raw_lines:
        obj = json.loads(line)

        key = (obj["model"], obj["question_id"], tuple(obj["judge"]))
        if key in visited:
            continue
        visited.add(key)

        model = obj["model"]
        if model not in selected_models:
            continue
        models.add(model)
        rets.append(obj)

    models = sorted(list(models))
    missing_models = [x for x in selected_models if x not in models]
    print(f"in models: {models}, number: {len(models)}")
    print(f"missing models: {missing_models}")
    print(f"#in: {len(raw_lines)}, #out: {len(rets)}")
    rets.sort(key=lambda x: (x["model"], x["question_id"], x["turn"]))

    with open(outfile, "w") as fout:
        for x in rets:
            fout.write(json.dumps(x) + "\n")
