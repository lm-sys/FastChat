import datetime
import glob
import json
from collections import deque
import tqdm


def _serialize_json(data):
    # Serialize JSON with sorted keys and no whitespace
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


types = {
    "share",
    "chat",
    "flag",
    "bothbad_vote",
    "downvote",
    "leftvote",
    "rightvote",
    "upvote",
    "tievote",
}

chat_dict = {}
cache_queue = deque()


def process_record(r):
    ip = r.pop("ip", None)
    tstamp = r.pop("tstamp")
    mtype = r.pop("type")
    start = r.pop("start", None)
    finish = r.pop("finish", None)

    # gabagge collect to save memory
    while len(cache_queue) > 100000:
        outdated = cache_queue.popleft()
        poped_item = chat_dict.pop(outdated["key"], None)
        if poped_item is None:
            # TODO: this sometimes happens, need to investigate what happens. in theory the chat dict should be synced with the queue, unless there are duplicated items
            print("Error: Key to GC does not exist.")

    assert mtype in types
    if mtype == "chat":
        key = _serialize_json(r["state"])
        # TODO: add the string length of the last reply for analyzing voting time per character.
        chat_dict[key] = {
            "timestamp": tstamp,
            "start": start,
            "finish": finish,
            "conv_id": r["state"]["conv_id"],
        }
        cache_queue.append({"key": key, "timestamp": tstamp})
    elif mtype in ("leftvote", "rightvote", "bothbad_vote", "tievote"):
        left_key = _serialize_json(r["states"][0])
        right_key = _serialize_json(r["states"][1])
        if left_key not in chat_dict:
            # TODO: this sometimes happens, it means we have the vote but we cannot find previous chat, need to investigate what happens
            print(
                f'WARNING: Cannot find vote context for conversation {r["states"][0]["conv_id"]}'
            )
            return
        if right_key not in chat_dict:
            print(
                f'WARNING: Cannot find vote context for conversation {r["states"][1]["conv_id"]}'
            )
            return
        vote_time_data = {
            "timestamp": tstamp,
            "type": mtype,
            "left": chat_dict[left_key],
            "right": chat_dict[right_key],
            "ip": ip,
        }
        return vote_time_data

    return None


def process_file(infile: str, outfile: str):
    with open(infile) as f:
        records = []
        for l in f.readlines():
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                    if r.get("tstamp") is not None:
                        records.append(r)
                except Exception:
                    pass
        # sort the record in case there are out-of-order records
        records.sort(key=lambda x: x["tstamp"])

        with open(outfile, "a") as outfile:
            for r in records:
                try:
                    output = process_record(r)
                    if output is not None:
                        outfile.write(json.dumps(output) + "\n")
                except Exception as e:
                    import traceback

                    print("Error:", e)
                    traceback.print_exc()


today = datetime.datetime.today().isoformat().split("T", 1)[0]
# sort it to make sure the date is continuous for each server
filelist = sorted(glob.glob("/mnt/disks/data/fastchat_logs/server*/202*-*-*-conv.json"))
filelist = [
    f for f in filelist if today not in f
]  # skip today because date could be partial

# TODO: change this to select different range of data
filelist = [f for f in filelist if "2024-03-" in f]

for f in tqdm.tqdm(filelist):
    process_file(f, "output.jsonl")
