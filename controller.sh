#/bin/bash

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

python3 -m fastchat.serve.controller --host 0.0.0.0
