FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# Add the deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.8.16
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils curl

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

# Install the required Python packages
RUN pip install prompt_toolkit rich safetensors fschat
