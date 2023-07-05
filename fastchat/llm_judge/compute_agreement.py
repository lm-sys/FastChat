"""
Compute agreement among judges.

Usage:
python compute_agreement.py --judges gpt4-pair human --votefiles human_judgments.json gpt4_pair_judgments.json
python compute_agreement.py --judges human human --votefiles human_judgments.json
"""
import argparse
import json
import os

import numpy as np


def get_judge_name(judge):
    if isinstance(judge, list) and judge[0] == "gpt-4" and judge[1].startswith("pair"):
        return "gpt4-pair"
    if judge.startswith("expert"):
        return "human"
    if judge.startswith("author"):
        return "author"


def revert(vote):
    if vote == "model_a":
        return "model_b"
    elif vote == "model_b":
        return "model_a"
    return vote


def get_mt_bench_votes_data(raw_votes):
    data = [{}, {}]

    for judge_votes in raw_votes:
        for vote in judge_votes:
            turn = vote["turn"] - 1
            if vote["model_a"] < vote["model_b"]:
                key = (vote["question_id"], vote["model_a"], vote["model_b"])
                winner = vote["winner"]
            else:
                key = (vote["question_id"], vote["model_b"], vote["model_a"])
                winner = revert(vote["winner"])
            judge = get_judge_name(vote["judge"])
            if key not in data[turn]:
                data[turn][key] = {}
            if judge not in data[turn][key]:
                data[turn][key][judge] = []
            data[turn][key][judge].append(winner)

    return data


def convertvote(vote):
    if "tie" in vote:
        return "tie"
    return vote


def equalvote(vote1, vote2):
    if "tie" in vote1 and "tie" in vote2:
        return True
    return vote1 == vote2


# data: Dict[qid -> List[vote]]
def get_mt_bench_agreement(data, judge1, judge2, ban):
    if judge1.startswith("gpt4") and judge2 == "human":
        stats = [0, 0]
        for votes in data.values():
            if judge1 not in votes or judge2 not in votes:
                continue
            assert len(votes[judge1]) == 1
            if convertvote(votes[judge1][0]) in ban:
                continue
            for v in votes[judge2]:
                if convertvote(v) in ban:
                    continue
                stats[1] += 1
                stats[0] += equalvote(votes[judge1][0], v)
        return stats[0], stats[1]
    elif judge1 == "human" and judge2 == "human":
        stats = [0, 0]
        for votes in data.values():
            if "human" not in votes:
                continue
            for i in range(len(votes["human"]) - 1):
                for j in range(i + 1, len(votes["human"])):
                    if (
                        convertvote(votes["human"][i]) in ban
                        or convertvote(votes["human"][j]) in ban
                    ):
                        continue
                    stats[1] += 1
                    stats[0] += equalvote(votes["human"][i], votes["human"][j])
        return stats[0], stats[1]
    else:
        raise Exception("Unsupported judges.")


def run_mt_bench_agreement(judges, votefiles):
    # votes[i]: List of votes
    votes = []
    for filename in votefiles:
        with open(filename, "r") as f:
            data = json.load(f)
        votes.append(data)

    data = get_mt_bench_votes_data(votes)

    agree, total = get_mt_bench_agreement(data[0], judges[0], judges[1], ban=[])
    print(
        f"turn 1 with tie. #total: {total}, #agree: {agree}, ratio: {agree/total:.2f}"
    )
    agree, total = get_mt_bench_agreement(data[0], judges[0], judges[1], ban=["tie"])
    print(
        f"turn 1 without tie. #total: {total}, #agree: {agree}, ratio: {agree/total:.2f}"
    )
    agree, total = get_mt_bench_agreement(data[1], judges[0], judges[1], ban=[])
    print(
        f"turn 2 with tie. #total: {total}, #agree: {agree}, ratio: {agree/total:.2f}"
    )
    agree, total = get_mt_bench_agreement(data[1], judges[0], judges[1], ban=["tie"])
    print(
        f"turn 2 without tie. #total: {total}, #agree: {agree}, ratio: {agree/total:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs=2, type=str, default=["gpt4-pair", "human"])
    parser.add_argument(
        "--votefiles",
        nargs="+",
        type=str,
        default=["gpt4_judgments.json", "human_judgments.json"],
    )
    args = parser.parse_args()

    run_mt_bench_agreement(args.judges, args.votefiles)
