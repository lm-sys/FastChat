FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl wget gnupg

# Add a non-root user
RUN useradd -m fastchat
USER fastchat
WORKDIR /home/fastchat

# Install Python and pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

# Install FastChat
RUN pip3 install fschat
RUN pip3 install fschat[model_worker,webui]

# Expose ports (if needed)
EXPOSE 8000
EXPOSE 21001

# Optional: Set an entrypoint or command
# ENTRYPOINT ["python3.9", "-m", "fastchat.serve.controller"]