#v2.4 with updates for the Vicuna fastchat install.  Added additional python packages for Gradio Web Interface, 
# moved python package install to a requirements.txt and added a Healthcheck component.
# Added Python_Cude_Alloc_Conf
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Metadata as described above
LABEL version="2.4" \
      description="Docker image for Vicuna fastchat with Gradio Web Interface" \
      maintainer="trig_tanning.Oj@icloud.com"

# Set environment variables
ENV PYTHONUNBUFFERED=1 
    # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256

# Add the deadsnakes PPA, install curl and required packages
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

# Copy requirements file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Add healthcheck (replace with your own healthcheck if not a web server on port 8000)
HEALTHCHECK --interval=5m --timeout=3s CMD curl -f http://localhost:8080/ || exit 1

