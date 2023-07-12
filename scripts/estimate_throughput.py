import argparse
import dataclasses


def compute_gpt_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False,
                       intermediate_size=None):
    """
    Compute the Tera Flop Operations (TFLOP) per second per GPU
    for GPT-like models.
    """
    factor = 2
    if backward:
        factor += 4
    if checkpoint_activations:
        factor += 2
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    total_flop = ((factor * num_layers * batch_size * seq_len * hidden_size *
                   (4 * hidden_size + 2 * intermediate_size + 2 * seq_len)) +
                  6 * batch_size * seq_len * hidden_size * vocab_size)
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops

@dataclasses.dataclass()
class Config:
    seq_len: int
    n_layers: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int


configs = {
    "llama-7b": Config(seq_len=2048, n_layers=32, hidden_size=4096, vocab_size=32000, intermediate_size=11008),
    "llama-30b": Config(seq_len=2048, n_layers=60, hidden_size=6656, vocab_size=32000, intermediate_size=17920)
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-7b")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-gpus", type=int, default=16)
    parser.add_argument("--latency", type=float, default=1.0)
    args = parser.parse_args()

    if args.model not in configs:
        raise RuntimeError("Unrecognized model")
    batch_size = args.batch_size
    seq_len = configs[args.model].seq_len
    n_layers = configs[args.model].n_layers
    hidden_size = configs[args.model].hidden_size
    vocab_size = configs[args.model].vocab_size
    intermediate_size = configs[args.model].intermediate_size

    tflops = compute_gpt_tflops(batch_size, seq_len, n_layers, hidden_size, vocab_size, args.n_gpus,
                       args.latency, checkpoint_activations=True, intermediate_size=intermediate_size)
    print(f"Model Flops: {tflops * 3 / 4}, HW Flops: {tflops}")
