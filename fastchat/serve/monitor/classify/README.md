## Download dataset
We have pre-generated several category classifier benchmarks and ground truths. You can download them (with [`git-lfs`](https://git-lfs.com) installed) to the directory `classify/` by running
```console
> git clone https://huggingface.co/datasets/lmarena-ai/categories-benchmark-eval
// cd into classify/ and then copy the label_bench directory to the current directory
> cp -r categories-benchmark-eval/label_bench . 
```
Your label_bench directory should follow the structure:
```markdown
├── label_bench/
│   ├── creative_writing_bench/
│   │   ├── data/
│   │   │    └── llama-v3p1-70b-instruct.json
│   │   └── test.json
│   ├── ...
│   ├── your_bench_name/
│   │   ├── data/
│   │   │    ├── your_classifier_data_1.json
│   │   │    ├── your_classifier_data_2.json
│   │   │    └── ...
│   │   └── test.json (your ground truth)
└── ...
```

## How to evaluate your category classifier?

To test your new classifier for a new category, you would have to make sure you created the category child class in `category.py`. Then, to generate classification labels, make the necessary edits in `config.yaml` and run
```console
python label.py --config config.yaml --testing
```

Then, add your new category bench to `tag_names` in `display_score.py`. After making sure that you also have a correctly formatted ground truth json file, you can report the performance of your classifier by running
```console
python display_score.py --bench <your_bench>
```

If you want to check out conflicts between your classifier and ground truth, use
```console
python display_score.py --bench <your_bench> --display-conflict
```

Example output:
```console
> python display_score.py --bench if_bench --display-conflict
Model: gpt-4o-mini-2024-07-18
Accuracy: 0.967
Precision: 0.684
Recall: 0.918

###### CONFLICT ######

Ground Truth = True; Pred = False
\####################
...

Ground Truth = False; Pred = True
\####################
...
```

