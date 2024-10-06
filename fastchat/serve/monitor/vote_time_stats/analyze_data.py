from collections import deque
import datetime
import glob
import hashlib
import json
from multiprocessing import Pool
import os

import tqdm


types = {'share', 'chat', 'flag', 'bothbad_vote', 'downvote', 'leftvote', 'rightvote', 'upvote', 'tievote'}

chat_dict = {}
cache_queue = deque()

conv_id_voted = set()


# See https://github.com/lm-sys/FastChat/blob/25062a1f317564057fd786b710d83ae66997a397/fastchat/serve/monitor/clean_battle_data.py#L95C1-L113C20
def replace_model_name(old_name, tstamp):
    replace_dict = {
        "bard": "palm-2",
        "claude-v1": "claude-1",
        "claude-instant-v1": "claude-instant-1",
        "oasst-sft-1-pythia-12b": "oasst-pythia-12b",
        "claude-2": "claude-2.0",
        "StripedHyena-Nous-7B": "stripedhyena-nous-7b",
        "gpt-4-turbo": "gpt-4-1106-preview",
        "gpt-4-0125-assistants-api": "gpt-4-turbo-browsing",
    }
    if old_name in ["gpt-4", "gpt-3.5-turbo"]:
        if tstamp > 1687849200:
            return old_name + "-0613"
        else:
            return old_name + "-0314"
    if old_name in replace_dict:
        return replace_dict[old_name]
    return old_name


def detect_language(text: str) -> str:
    """Detect the langauge of a string."""
    import polyglot  # pip3 install polyglot pyicu pycld2
    from polyglot.detect import Detector
    from polyglot.detect.base import logger as polyglot_logger
    import pycld2

    polyglot_logger.setLevel("ERROR")

    try:
        lang_code = Detector(text).language.name
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        lang_code = "unknown"
    return lang_code


def preprocess_record(r: dict):
    # Features
    # chat: state, battle: states[0] & states[1]
    ip = r.pop("ip", "")
    tstamp = r.pop("tstamp")
    mtype = r.pop("type")
    start = r.pop("start", None)
    finish = r.pop("finish", None)

    if tstamp is None:
        # skip invalid data point
        return

    assert mtype in types
    if mtype == "chat":
        key = state2key(r["state"])
        last_message = r["state"]["messages"][-1][-1]
        if last_message is None or last_message.strip() == "":
            return
        return {
            "timestamp": tstamp,
            "type": mtype,
            "start": start,
            "finish": finish,
            "conv_id": r["state"]["conv_id"],
            "last_message_len": len(last_message),
            "key": key,
        }
    elif mtype in ("leftvote", "rightvote", "bothbad_vote", "tievote") and r.get("models") == ["", ""]:
        left_state = r["states"][0]
        right_state = r["states"][1]

        # Detect langauge
        if left_state["offset"] >= len(left_state["messages"]):
            return
        lang_code = detect_language(left_state["messages"][left_state["offset"]][1])

        left_key = state2key(left_state)
        right_key = state2key(right_state)
        return {
            "timestamp": tstamp,
            "type": mtype,
            "keys": (left_key, right_key),
            "vote_id": (left_state["conv_id"], right_state["conv_id"]),
            "models": [
                replace_model_name(left_state["model_name"], tstamp),
                replace_model_name(right_state["model_name"], tstamp)
            ],
            "ip": ip,
            "language": lang_code,
        }
    return None


def read_jsonlines_chunk(args):
    """Read a chunk of a JSONLines file and return the results."""
    file_path, start, end = args
    results = []
    with open(file_path, 'r') as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline().strip()
            if line:
                try:
                    r = json.loads(line)
                    output = preprocess_record(r)
                    if output is not None:
                        results.append(output)
                except Exception:
                    pass
    # We sort here first, because Python is using Timsort which is efficient for partially sorted lists.
    results.sort(key=lambda x: x["timestamp"])
    return results


def chunkify(file_path: str, num_chunks: int) -> list:
    """Split the file into chunks based on the number of processes."""
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_chunks
    chunks = []

    with open(file_path, 'r') as f:
        start = 0
        while start < file_size:
            end = min(start + chunk_size, file_size)
            f.seek(end)
            f.readline()  # Ensure we read to the end of the line
            end = f.tell()
            chunks.append((file_path, start, end))
            start = end

    return chunks


def parallel_read_jsonlines(file_path: str, num_chunks: int, pool):
    """Read a JSONLines file in parallel using multiple processes."""
    chunks = chunkify(file_path, num_chunks=num_chunks)
    results = pool.map(read_jsonlines_chunk, chunks)

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]
    flattened_results.sort(key=lambda x: x["timestamp"])
    return flattened_results


def _serialize_json(data):
    # Serialize JSON with sorted keys and no whitespace
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')


def state2key(state: dict) -> str:
    # NOTE: we use md5 hash here to accelerate the process, assuming the context is not adversarial.
    return hashlib.md5(_serialize_json(state)).hexdigest()


def process_record(r):
    global chat_dict, cache_queue

    # gabagge collect to save memory
    while len(cache_queue) > 100000:
        outdated = cache_queue.popleft()
        p = chat_dict.pop(outdated["key"], None)
        if p is None:
            print("Error: Key to GC does not exist.", len(chat_dict), len(cache_queue))

    mtype = r["type"]

    if mtype == "chat":
        key = r.pop("key")
        if key in chat_dict:
            # TODO: figure out why there are duplicates. regenerate? concurrency issue?
            return
        chat_dict[key] = r
        cache_queue.append({"key": key, "timestamp": r["timestamp"]})
    elif mtype in ("leftvote", "rightvote", "bothbad_vote", "tievote"):
        keys = r.pop("keys")
        vote_id = r.pop("vote_id")

        if keys[0] not in chat_dict:
            print(f'WARNING: Cannot find vote context for conversation {vote_id[0]}')
            return
        if keys[1] not in chat_dict:
            print(f'WARNING: Cannot find vote context for conversation {vote_id[1]}')
            return
        left_chat = chat_dict[keys[0]]
        right_chat = chat_dict[keys[1]]

        if vote_id in conv_id_voted:
            # the conversation has been voted. the following votes should not be considered
            # TODO: is this expected?
            return
        conv_id_voted.add(vote_id)

        r["left"] = left_chat
        r["right"] = right_chat
        return r

    return None


def process_files(filelist: list, num_processes: int, output_file: str):
    with Pool(num_processes) as pool:
        for f in tqdm.tqdm(filelist, desc="Processing"):
            results = parallel_read_jsonlines(f, num_processes, pool)
            with open(output_file, "a") as outfile:
                for r in results:
                    try:
                        output = process_record(r)
                        if output is not None:
                            outfile.write(json.dumps(output) + "\n")
                    except Exception as e:
                        import traceback
                        print("Error:", e)
                        traceback.print_exc()


if __name__ == "__main__":
    today = datetime.datetime.today().isoformat().split("T", 1)[0]
    filelist = sorted(glob.glob("/mnt/disks/data/fastchat_logs/server*/202*-*-*-conv.json"))
    filelist = [f for f in filelist if today not in f]  # skip today because data could be partial

    selected_month = "2024-03"
    filelist = [f for f in filelist if selected_month in f]  # select month to process

    process_files(filelist, num_processes=48, output_file=f"votes_{selected_month}.jsonl")
