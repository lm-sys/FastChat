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
python3 elo_analysis.py --clean-battle-file clean_battle_20230523.json
```
