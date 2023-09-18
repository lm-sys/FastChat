"""Count the unique users in a battle log file."""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()

    lines = json.load(open(args.input))
    ct_anony_votes = 0
    all_users = set()
    all_models = set()
    for l in lines:
        if not l["anony"]:
            continue
        all_users.add(l["judge"])
        all_models.add(l["model_a"])
        all_models.add(l["model_b"])
        ct_anony_votes += 1

    print(f"#anony_vote: {ct_anony_votes}, #user: {len(all_users)}")
    print(f"#model: {len(all_models)}")
