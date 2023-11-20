import json
import os

import tqdm


def get_all_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        filenames.append(filename)
    return filenames
sorce_index={}
directory = "/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/data/processed/"  # 请替换为你需要获取文件名的目录路径
all_filenames = get_all_filenames(directory)

for file in tqdm.tqdm(all_filenames):
    data=json.load(open(directory+file))
    sorce_cnt=0 # 有几项分数
    sorce=0
    try:
        if "可懂性" in data.keys():
            sorce+=int(data["可懂性"])
            sorce_cnt+=1
    except:
        print(data["可懂性"])
    try:
        if "正确性" in data.keys():
            sorce+=int(data["正确性"])
            sorce_cnt+=1
    except:
        print(data["正确性"])
    try:
        if "信息量" in data.keys():
            sorce+=int(data["信息量"])
            sorce_cnt+=1
    except:
        print(data["信息量"])
    try:
        if "confidence" in data.keys():
            sorce+=int(data["confidence"])
            sorce_cnt+=1
    except:
        print(data["confidence"])
    if sorce_cnt!=0:
        sorce = sorce/sorce_cnt
    sorce_index.update({data["index"]:sorce})
with open("/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/data/sorce_index.json","a") as f:
    json.dump(sorce_index,f,ensure_ascii=False)
print(len(sorce_index))
