import argparse
import sky

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--controller-name", type=str, default="fastchat-controller")
    parser.add_argument("--worker-name", type=str, default="gpu-worker")
    
    args = parser.parse_args()
    if len(sky.status(args.controller_name)) == 0:
        task = sky.Task.from_yaml("controller.yaml")
        sky.launch(task, cluster_name=args.controller_name)
        task = sky.Task.from_yaml("gradio.yaml")
        sky.exec(task, cluster_name=args.controller_name)

    task = sky.Task.from_yaml("model_worker.yaml")
    head_ip = sky.status(args.controller_name)[0]['handle'].head_ip
    envs = {"CONTROLLER_IP": head_ip}
    task.update_envs(envs)
    
    for i in range(args.num_gpus):
        worker_name = f"{args.worker_name}-{i}"
        if args.spot:
            sky.spot_launch(task, name=worker_name)
        else:
            sky.launch(task, cluster_name=worker_name, detach_setup=True)