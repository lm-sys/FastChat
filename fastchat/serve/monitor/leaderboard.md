## elo ratings (bootstrap scores - MLE)
|    | model                   |    score |    lower |    upper |   error_y_minus |   error_y |
|---:|:------------------------|---------:|---------:|---------:|----------------:|----------:|
|  0 | vicuna-13b              | 1226.64  | 1105.87  | 1351.8   |         120.771 |   125.154 |
|  1 | koala-13b               | 1104.7   |  950.628 | 1254.12  |         154.072 |   149.421 |
|  2 | alpaca-13b              |  976.923 |  830.847 | 1103.49  |         146.077 |   126.568 |
|  3 | chatglm-6b              |  938.927 |  803.489 | 1078.78  |         135.438 |   139.856 |
|  4 | oasst-pythia-12b        |  896.972 |  782.219 | 1024.56  |         114.754 |   127.59  |
|  5 | stablelm-tuned-alpha-7b |  755.621 |  609.506 |  886.191 |         146.115 |   130.57  |
|  6 | dolly-v2-12b            |  716.604 |  592.414 |  869.856 |         124.19  |   153.252 |
|  7 | llama-13b               |  581.831 |  444.675 |  711.009 |         137.156 |   129.178 |
## elo ratings (bootstrap scores - linear update)
|    | model                   |    score |    lower |    upper |   error_y_minus |   error_y |
|---:|:------------------------|---------:|---------:|---------:|----------------:|----------:|
|  0 | vicuna-13b              | 1190.25  | 1135.6   | 1244.41  |         54.6463 |   54.1572 |
|  4 | koala-13b               | 1087.87  | 1038.29  | 1143.4   |         49.5739 |   55.5384 |
|  1 | alpaca-13b              | 1038     |  967.536 | 1103.41  |         70.4636 |   65.4127 |
|  3 | oasst-pythia-12b        | 1010.88  |  959.584 | 1066.45  |         51.2983 |   55.568  |
|  7 | chatglm-6b              |  941.48  |  884.43  | 1013.63  |         57.0499 |   72.1467 |
|  6 | stablelm-tuned-alpha-7b |  931.93  |  884.401 | 1007.09  |         47.5294 |   75.1586 |
|  2 | dolly-v2-12b            |  904.636 |  846.2   |  973.263 |         58.4364 |   68.6274 |
|  5 | llama-13b               |  893.508 |  838.28  |  964.056 |         55.2275 |   70.548  |

# elo rating algorithm (MLE) description - Joey?
TODO

# elo rating algorithm (linear update) description

https://medium.com/purple-theory/what-is-elo-rating-c4eb7a9061e0

If players A and B have ratings R_A and R_B, then the expected scores are given by
E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
E_B = 1 / (1 + 10 ^ ((R_A - R_B) / 400))

Linear update:

R_A' = R_A + K(S_A - E_A)

S_A is the outcome (1 for win, 0 for lose)

Notes:
A player’s expected score = their probability of winning + half their probability of drawing.

TODO
