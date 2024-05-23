import argparse
import json

# See https://github.com/lm-sys/FastChat/blob/25062a1f317564057fd786b710d83ae66997a397/fastchat/serve/monitor/clean_battle_data.py#L140C2-L314C16

def preprocess_record(r: dict, minimum_voting_time: float):
    convert_type = {
        "leftvote": "model_a",
        "rightvote": "model_b",
        "tievote": "tie",
        "bothbad_vote": "tie (bothbad)",
    }
    user_id = r["ip"]
  
    time_to_vote = r["timestamp"] - max(r["left"]["finish"], r["right"]["finish"])
    if time_to_vote < minimum_voting_time:
        return None

    return dict(
        question_id=r["left"]["conv_id"],
        model_a=r["models"][0],
        model_b=r["models"][1],
        winner=convert_type[r["type"]],
        judge=f"arena_user_{user_id}",
        conversation_a=None,
        conversation_b=None,
        turn=0,
        anony=True,
        language=r["language"],
        tstamp=r["timestamp"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean battle data")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file", default=None)
    parser.add_argument("--minimum-voting-time", "-m", type=float, help="The minimum time required for voting", default=0.0)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".jsonl", ".battle.jsonl")

    with open(args.output, "w") as output:
        with open(args.input, "r") as input:
            for line in input:
                record = json.loads(line)
                cleaned_record = preprocess_record(record, args.minimum_voting_time)
                if cleaned_record is not None:
                    output.write(json.dumps(cleaned_record) + "\n")
