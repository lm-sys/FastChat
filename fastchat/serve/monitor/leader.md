## elo ratings (bootstrap scores - MLE)
|    | model                   |    score |    lower |    upper |   error_y_minus |   error_y |
|---:|:------------------------|---------:|---------:|---------:|----------------:|----------:|
|  0 | vicuna-13b              | 1203.92  | 1142.72  | 1293.29  |         61.1984 |   89.3697 |
|  1 | koala-13b               | 1051.47  |  966.335 | 1169.67  |         85.1371 |  118.202  |
|  2 | alpaca-13b              |  956.341 |  842.568 | 1086.32  |        113.773  |  129.977  |
|  3 | chatglm-6b              |  889.133 |  773.402 | 1019.13  |        115.731  |  129.998  |
|  4 | oasst-pythia-12b        |  868.516 |  740.666 |  970.714 |        127.85   |  102.198  |
|  5 | stablelm-tuned-alpha-7b |  737.95  |  593.235 |  887.349 |        144.714  |  149.399  |
|  6 | dolly-v2-12b            |  655.17  |  533.494 |  781.606 |        121.676  |  126.436  |
|  7 | llama-13b               |  604.358 |  483.598 |  700.627 |        120.76   |   96.269  |
## elo ratings (bootstrap scores - linear update)
|    | model                   |    score |    lower |    upper |   error_y_minus |   error_y |
|---:|:------------------------|---------:|---------:|---------:|----------------:|----------:|
|  0 | vicuna-13b              | 1186.01  | 1105.95  | 1253.92  |         80.0554 |   67.9136 |
|  4 | koala-13b               | 1102.8   | 1007.82  | 1175.67  |         94.9829 |   72.874  |
|  1 | alpaca-13b              | 1028.2   |  961.885 | 1119.68  |         66.3119 |   91.4845 |
|  3 | oasst-pythia-12b        | 1006.67  |  920.344 | 1101.28  |         86.3275 |   94.6094 |
|  7 | chatglm-6b              |  975.749 |  873.1   | 1065.72  |        102.649  |   89.9737 |
|  6 | stablelm-tuned-alpha-7b |  923.46  |  836.861 | 1028.64  |         86.5989 |  105.176  |
|  2 | dolly-v2-12b            |  911.547 |  832.537 |  993.289 |         79.01   |   81.7419 |
|  5 | llama-13b               |  860.979 |  789.803 |  954.857 |         71.1761 |   93.8783 |

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
