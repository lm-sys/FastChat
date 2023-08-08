from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import os
import time
from collections import OrderedDict
from typing import Optional, List
import io
import subprocess


import torch
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import AutoConfig
from google.cloud import storage


from config import CONFIG_LOCATION, load_tokenizer
from subclass import YieldingLlama
import logging

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"

def test_cli_deserialization():

    path = "path_to_weights"

    st = time.time()
    print("downloading weights")
    # don't love the whole /gc/google-cloud-sdk/bin/gcloud path but don't think there's an easy way to update PATH from cog, so might as well do this.
    weights = "/src/llama_tensors"
    os.system(f"/gc/google-cloud-sdk/bin/gcloud storage cp {path} {weights}")
    print(f"weignts downloaded in {time.time() - st}")
    print(f"deserializing weights from {weights}")
    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: YieldingLlama.from_pretrained(
            None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
        )
    )
    logging.disable(logging.NOTSET)
    des = TensorDeserializer(weights, plaid_mode=False)
    des.load_into_module(model)
    print(f"zero to weights in {time.time() - st}")


def test_in_memory_cli_deserialization():
    """This is quite slow, turns out that gcloud storage streaming into memory (-) runs in series."""
    path = "path/to/weights"
    st = time.time()
    print("downloading weights")
    # don't love the whole /gc/google-cloud-sdk/bin/gcloud path but don't think there's an easy way to update PATH from cog, so might as well do this.
    command = f"/gc/google-cloud-sdk/bin/gcloud storage cp {path} -".split()
    result = subprocess.run(command, stdout=subprocess.PIPE, text=False)
    if result.returncode != 0:
        raise Exception(f"gcloud storage cp command failed with return code {result.returncode}: {result.stderr.decode('utf-8')}")

    in_memory_file = io.BytesIO(result.stdout)
    in_memory_file.seek(0)

    print(f"weignts downloaded in {time.time() - st}")
    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

    logging.disable(logging.WARN)
    model = no_init_or_tensor(
        lambda: YieldingLlama.from_pretrained(
            None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
        )
    )
    logging.disable(logging.NOTSET)
    des = TensorDeserializer(in_memory_file, plaid_mode=False)
    des.load_into_module(model)
    print(f"zero to weights in {time.time() - st}")


def download_chunk(dl_cfg):
    """Submittable function to python process pool for downloading byte chunk"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(dl_cfg['bucket'])
    blob = bucket.get_blob(dl_cfg['blob'])
    in_memory_file = io.BytesIO()
    blob.download_to_file(in_memory_file, start=dl_cfg['start'], end=dl_cfg['end'])
    return in_memory_file

def download_blob_to_stream(bucket_name: str, source_blob_name: str, n: int = 4):
    """Downloads a blob to a stream or other file-like object."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Need to call get_blob to get metadata
    blob = bucket.get_blob(source_blob_name)

    def _partition_file(size: int, bucket: str, blob: str, n: int) -> List[dict]:
        partitions = []
        split = int(size/n)
        start = 0
        end = split

        for i in range(n):
            if i == n - 1:  # If it's the last partition
                end = size - 1  # Set the endpoint to the last byte of the file
            partitions.append({"start": start, "end": end, "bucket": bucket, "blob": blob})
            start = end + 1
            end += split

        return partitions
    
    partitions = _partition_file(blob.size, bucket_name, source_blob_name, n)

    print('submitting tasks')
    res = []
    with ProcessPoolExecutor(n) as ex:
        res = list(ex.map(download_chunk, partitions))
        # results = [ex.submit(download_chunk, partition) for partition in partitions]
        
        # for future in concurrent.futures.as_completed(results):
        #     res.append(future.result())
    print('all downloads finished')

    concatenated_bytes = b''.join(result.getvalue() for result in res)

    # Create a new in memory file w/all bytes concatenated
    in_memory_file = io.BytesIO(concatenated_bytes)
    in_memory_file.seek(0)
    return in_memory_file


def test_python_deserialization():
    st = time.time()
    print("downloading weights")
    bucket_name = "CHANGEME"
    source_name = "CHANGEME"

    obj = download_blob_to_stream(bucket_name=bucket_name, source_blob_name=source_name, n=24)

    print(f"weignts downloaded in {time.time() - st}")

    print(f"deserializing weights from memory")
    config = AutoConfig.from_pretrained(CONFIG_LOCATION)

    logging.disable(logging.WARN) # turns off long message about not training the model
    model = no_init_or_tensor(
        lambda: YieldingLlama.from_pretrained(
            None, config=config, state_dict=OrderedDict()
        )
    )
    logging.disable(logging.NOTSET)
    des = TensorDeserializer(obj, plaid_mode=False)
    des.load_into_module(model)
    print(f"zero to weights in {time.time() - st}")


if __name__ == '__main__':
    #test_python_deserialization()
    test_cli_deserialization()