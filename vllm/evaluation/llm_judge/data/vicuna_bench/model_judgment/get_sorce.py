import json
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

judgement_file = "gpt-3.5-turbo_single.jsonl"
judgement = []
with open(judgement_file, mode="r", encoding="utf-8") as fr:
    for j in fr.readlines():
        a = json.loads(j)
        judgement.append(a)

model_list = ["llama-2", "l_alpaca-203", "l_alpaca-406", "l_alpaca-609", "l_alpaca-812", "llama-2-chat"]
model_score = {}
for model in model_list:
    model_score[model] = []

for j in judgement:
    score = j["score"]
    model_score[j["model"]].append(score)

for model, score in model_score.items():
    mean_score = sum(score)/len(score)
    model_score[model].append(mean_score)

score = []
model_id = ["llama-2", "alpaca", "llama-2-chat"]
model_list1 = ["llama-2", "l_alpaca-609", "llama-2-chat"]
for m in model_list:
    if m in model_list1:
        score.append(model_score[m][-1])
x = range(len(model_id))
print(score)
plt.bar(x, score)
plt.title("vicuna_bench上的分数比较")
plt.xlabel("模型名称")
plt.xticks(x,model_id)
plt.ylabel("分数")
plt.grid(True)
plt.show()


