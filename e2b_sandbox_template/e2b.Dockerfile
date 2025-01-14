# e2b base image
FROM --platform=linux/x86_64 e2bdev/code-interpreter:latest
# FROM node:21-slim

# Install Python
RUN apt-get update && apt-get install -y python3

# Pre-Install Python packages
RUN pip install uv
RUN uv pip install --system pandas numpy matplotlib requests seaborn plotly
RUN uv pip install --system pygame pygbag black
RUN uv pip install --system --upgrade streamlit gradio nicegui

# Install nginx
RUN apt-get update && apt-get install -y nginx
# Add Nginx configuration and serve with Nginx
COPY nginx/nginx.conf /etc/nginx/sites-enabled/default
# CMD ["nginx", "-g", "daemon off;"]

# Build container_app
WORKDIR /home/user/container_app
COPY container_app/ ./
RUN npm install
RUN npm run build

# Build react app
WORKDIR /home/user/react_app
COPY react_app/ ./
RUN npm install
RUN npm run build

# Build vue app
WORKDIR /home/user/vue_app
COPY vue_app/ ./
RUN npm install
RUN npm run build

# Prepare Python app
WORKDIR /home/user/python_app
COPY python_app/ ./