# Use the official Python image as the parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install git+https://github.com/huggingface/transformers

# Clone the FastChat repository
RUN git clone https://github.com/lm-sys/FastChat.git

# Set the working directory to /app/FastChat
WORKDIR /app/FastChat

# Install the package
RUN pip3 install -e .

# Download the Vicuna-13B weights
RUN python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-13b \
    --target /app/FastChat/fastchat/models/vicuna-13b \
    --delta lmsys/vicuna-13b-delta-v0

# Set the working directory to /app/FastChat/fastchat/serve
WORKDIR /app/FastChat/fastchat/serve

# Expose the port
EXPOSE 8000

# Start the server
CMD ["python3", "-m", "fastchat.serve.cli", "--model-name", "/app/FastChat/fastchat/models/vicuna-13b"]
