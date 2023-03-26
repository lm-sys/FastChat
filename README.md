# ChatServer
A chatbot server.

## Install

### Method 1: From Source
```
pip3 install -e .

# Install the main branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers
```

## Serving

### Web UI
```
# Launch a controller
python3 -m chatserver.serve.controller

# Launch a model worker
python3 -m chatserver.serve.model_worker --model facebook/opt-350m

# Send a test message
python3 -m chatserver.serve.test_message

# Luanch a gradio web server.
python3 -m chatserver.serve.gradio_web_server

# You can open your brower and chat with a model now.
```

### Command Line Interface
```
python3 -m chatserver.serve.cli --model facebook/opt-350m
```

## Deploy Chatbot on Any Cloud with SkyPilot
### Training Vicuna with SkyPilot
1. Install skypilot and setup the credentials locally following the instructions [here](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)
```
# Need this version of skypilot, for the fix of `--env` flag.
pip install git+https://github.com/skypilot-org/skypilot.git
```
2. Train the model
```
sky launch -c vicuna -s scripts/train-vicuna.yaml --env WANDB_API_KEY

# Launch it on managed spot
sky spot launch -n vicuna scripts/train-vicuna.yaml --env WANDB_API_KEY

# Train a 7B model
sky launch -c vicuna -s scripts/train-vicuna.yaml --env WANDB_API_KEY --env MODEL_SIZE=7
```

### Training Alpaca with SkyPilot
Launch the training job with the following line (will be launched on a single node with 4 A100-80GB GPUs)
```
# WANDB API KEY is required for logging. We use the key in your local environment.
sky launch -c alpaca -s scripts/train-alpaca.yaml --env WANDB_API_KEY

# Train the 13B model
sky launch -c alpaca -s scripts/train-alpaca.yaml --env WANDB_API_KEY --env MODEL_SIZE=13

# You can use a manged spot instance.
sky spot launch -n alpaca scripts/train-alpaca.yaml --env WANDB_API_KEY
```

### Serving Alpaca with SkyPilot
1. We assume SkyPilot is installed and the model checkpoint is stored on some cloud storage (e.g., GCS).
2. Launch the controller server (default to a cheap CPU VM):
    ```
    sky launch -c controller scripts/serving/controller.yaml
    ```
3. Find the IP address of the controller server on the cloud console. Make sure the ports are open (default port 21001 for controller, 21002 for model workers).
4. Launch a model worker (default to A100):
    ```
    sky launch -c model-worker scripts/serving/model_worker.yaml --env CONTROLLER_IP=<controller-ip>
    ```
    You can use spot instances to save 3x cost. SkyPilot will automatically recover the spot instance if it is preempted ([more details](https://skypilot.readthedocs.io/en/latest/examples/spot-jobs.html)).
    ```
    sky spot launch scripts/serving/model_worker.yaml --env CONTROLLER_IP=<controller-ip>
    ```
5. Click the link generated from step 2 and chat with AI :)
![screenshot](./assets/screenshot.png)
