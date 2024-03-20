<<<<<<< HEAD
# v2.8 with updates for the Vicuna fastchat install. Added additional python packages for Gradio Web Interface,
# moved python package install to a requirements.txt and added a Healthcheck component.
# Added Python_Cude_Alloc_Conf, and path for use of cuda. Added signature and other ignores for ubuntu and nvidia repo errors
# CDN bug opened for nvidia.

# Base image with CUDA support
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Metadata as described above
LABEL version="2.8" \
      description="Docker image for Vicuna fastchat with Gradio Web Interface" \
      maintainer="trig_tanning.Oj@icloud.com"

# Set environment variables, CUDA max split to help with memory crash issue.
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/cuda/bin:$PATH
    # Note: If you need the PYTORCH_CUDA_ALLOC_CONF, uncomment the next line
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Get GPG Keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# Clear APT cache and allow apt to continue even if it encounters signature errors
RUN apt-get clean && apt-get update --allow-insecure-repositories

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Add the repository directly to resolve signature issues during docker build, added git to install
RUN echo "deb [trusted=yes] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" >> /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    apt-get update --allow-insecure-repositories && \
    apt-get install -y software-properties-common curl python3.8 python3.8-distutils git && \  
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py


# Clear pip cache
RUN pip cache purge

# Copy requirements file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt && rm -rf /root/.cache/pip/*

# Check nvcc access
# RUN which nvcc

# Install Flash-attn for training, requires ninja, packaging, pytorch. Currently commented out to get environment up.
# RUN pip install flash-attn>=2.0

# Add healthcheck (replace with your own healthcheck if not a web server on port 8080)
HEALTHCHECK --interval=5m --timeout=3s CMD curl -f http://localhost:8080/ || exit 1
=======
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
RUN pip3 install fschat
RUN pip3 install fschat[model_worker,webui] pydantic==1.10.13
>>>>>>> main
