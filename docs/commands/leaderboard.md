### Get logs
```
gsutil -m rsync -r gs://fastchat_logs ~/fastchat_logs/
```

### Clean battle data
```
cd ~/FastChat/fastchat/serve/monitor
python3 clean_battle_data.py
```

### Run Elo analysis
```
python3 elo_analysis.py --clean-battle-file clean_battle_20230905.json
```

### Copy files to HF space
1. update plots
```
scp atlas:/data/lmzheng/FastChat/fastchat/serve/monitor/elo_results_20230905.pkl .
```

2. update table
```
wget https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/raw/main/leaderboard_table_20230905.csv
```

### Update files on webserver
```
DATE=20231002

rm -rf elo_results.pkl leaderboard_table.csv
wget https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/elo_results_$DATE.pkl
wget https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/resolve/main/leaderboard_table_$DATE.csv
ln -s leaderboard_table_$DATE.csv leaderboard_table.csv
ln -s elo_results_$DATE.pkl elo_results.pkl
```
