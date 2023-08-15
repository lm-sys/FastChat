#v2.1 with updates for the Vicuna fastchat install.  Added additional python packages for Gradio Web Interface
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set environment variable to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Add the deadsnakes PPA and install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

# Install the required Python packages
RUN pip install numpy==1.24.2 \
    torch==2.0.1 \
    accelerate==0.20.3 \
    peft==0.2.0 \
    sentencepiece==0.1.97 \
    tensorizer==1.0.1 \
    jinja2==3.1.2 \
    deepspeed==0.8.3 \
    "transformers>=4.31" \
    prompt_toolkit==3.0.39 \
    "markdown-it-py<=3.0.0" \
    mdurl==0.1.2 \
    pygments==2.16.1 \
    "rich<=13.5.2" \
    safetensors==0.3.2 \
    gradio==3.40 \
    "plotly<=5.16.0" \
    fschat
