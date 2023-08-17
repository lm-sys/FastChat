def add_launch_all_serve_args():
    # ------multi worker-----------------
    parser.add_argument(
        "--model-path-address",
        default="THUDM/chatglm2-6b@localhost@20002",
        nargs="+",
        type=str,
        help="model path, host, and port, formatted as model-path@host@port",
    )
    # ---------------controller-------------------------
    parser.add_argument("--controller-host", type=str, default="localhost")
    parser.add_argument("--controller-port", type=int, default=21001)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    # ----------------------worker------------------------------------------
    parser.add_argument("--worker-host", type=str, default="localhost")
    parser.add_argument("--worker-port", type=int, default=21002)
    # parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    # parser.add_argument(
    #     "--controller-address", type=str, default="http://localhost:21001"
    # )
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.3",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Load quantized model. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    # -----------------openai server---------------------------
    parser.add_argument("--server-host", type=str, default="localhost", help="host name")
    parser.add_argument("--server-port", type=int, default=8001, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    # parser.add_argument(
    #     "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    # )
    # parser.add_argument(
    #     "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    # )
    # parser.add_argument(
    #     "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    # )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )