# Elo
https://medium.com/purple-theory/what-is-elo-rating-c4eb7a9061e0

If players A and B have ratings R_A and R_B, then the expected scores are given by
E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
E_B = 1 / (1 + 10 ^ ((R_A - R_B) / 400))

Linear update:

R_A' = R_A + K(S_A - E_A)

S_A is the outcome (1 for win, 0 for lose)

Notes:
A player’s expected score = their probability of winning + half their probability of drawing.

