#v2.7 with updates for the Vicuna fastchat install.  Added additional python packages for Gradio Web Interface, 
# moved python package install to a requirements.txt and added a Healthcheck component.
# Added Python_Cude_Alloc_Conf, and path for use of cuda.

# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Metadata as described above
LABEL version="2.4" \
      description="Docker image for Vicuna fastchat with Gradio Web Interface" \
      maintainer="trig_tanning.Oj@icloud.com"

# Set environment variables, CUDA max split to help with memory crash issue.
ENV PYTHONUNBUFFERED=1 \
    PATH=/usr/local/cuda/bin:$PATH
    # Note: If you need the PYTORCH_CUDA_ALLOC_CONF, uncomment the next line
    # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256

# Claer APT cache and allow apt to continue even if it encounters signature errors
RUN apt-get clean && apt-get update --allow-insecure-repositories

# Add the deadsnakes PPA, install curl and required packages
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils && \
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
RUN which nvcc

# Install Flash-attn for training, requires ninja, packaging, pytorch. Currently commented out to get environment up.
RUN pip install flash-attn>=2.0

# Add healthcheck (replace with your own healthcheck if not a web server on port 8080)
HEALTHCHECK --interval=5m --timeout=3s CMD curl -f http://localhost:8080/ || exit 1
