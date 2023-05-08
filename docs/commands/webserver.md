### Install
```
sudo apt update
sudo apt install tmux htop

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh

conda create -n fastchat python=3.9
conda activate fastchat

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install -e .

# Install the latest main branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers
```

### Launch servers
```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001
python3 -m fastchat.serve.register_worker --controller http://localhost:21001 --worker-name https://
python3 -m fastchat.serve.test_message --model vicuna-13b --controller http://localhost:21001

export OPENAI_API_KEY=
python3 -m fastchat.serve.gradio_web_server --controller http://localhost:21001 --moderate --concurrency 20
```

### Increase the limit of max open files
One process (do not need reboot)
```
sudo prlimit --nofile=1048576:1048576 --pid=$id

for id in $(ps -ef | grep gradio_web_server | awk '{print $2}'); do echo $id; prlimit --nofile=1048576:1048576 --pid=$id; done
```

System (need reboot): Add the lines below to `/etc/security/limits.conf`
```
* hard nofile 65535
* soft nofile 65535
```
