# Instructions

First run `analyze_data.py` to collect metadata of all votes.

Then run `plot.py` to get the plot. You need to edit these files for proper input or output filename

python clean_battle.py -i votes_2024-04.jsonl -o votes_2024-04.battle.jsonl -m 0.0
python clean_battle.py -i votes_2024-04.jsonl -o votes_2024-04.battle_120.jsonl -m 120.0
python clean_battle.py -i votes_2024-04.jsonl -o votes_2024-04.battle_60c.jsonl -c 0.6

python compute_elo.py -i votes_2024-04.battle.jsonl -o votes_2024-04.elo.jsonl
python compute_elo.py -i votes_2024-04.battle_120.jsonl -o votes_2024-04.elo_120.jsonl
python compute_elo.py -i votes_2024-04.battle_c60.jsonl
