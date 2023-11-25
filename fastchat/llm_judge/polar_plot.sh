#!/bin/bash

python polar_plot.py --cfg_fn ./data/mt_bench/polar_plot_configs/gpt4-judge-all.json
python polar_plot.py --cfg_fn ./data/mt_bench/polar_plot_configs/gpt4-judge-gpt.json
python polar_plot.py --cfg_fn ./data/mt_bench/polar_plot_configs/gpt4-turbo-judge.json