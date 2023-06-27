#/bin/bash

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

#python3 -m fastchat.serve.gradio_web_server_multi --host 0.0.0.0
#python3 -m fastchat.serve.gradio_web_server --model-list-mode=reload --host 0.0.0.0
python3 fastchat/serve/gradio_web_server_branded.py --model-list-mode=reload --host 0.0.0.0
