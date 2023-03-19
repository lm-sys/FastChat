# ChatServer
chatbot server


## Train Alpaca with SkyPilot
1. Install skypilot and setup the credentials locally following the instructions [here](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)
2. Launch the training job with the following line (will be launched on a single node with 4 A100-80GB GPUs)
    ```
    # WANDB API KEY is required for logging. We use the key in your local environment.
    sky launch -c alpaca -s scripts/train.yaml --env WANDB_API_KEY
    ```
    Or use spot (not managed).
    ```
    sky launch -c alpaca-spot -s --use-spot scripts/train.yaml --env WANDB_API_KEY
    ```
    **The following still does not work at the moment as Alpaca code does not support multiple nodes.**
    We can also launch the training job with multiple nodes and different number of GPUs. We will automatically adapt the
    gradient accumulation steps to the setting (Supported max number of #nodes * #GPUs per node = 32)
    ```
    sky launch -c alpaca-2 -s --num-nodes 2 --gpus A100-80GB:8 scripts/train.yaml  --env WANDB_API_KEY
    ```
    Managed spot version TO BE ADDED.


