import subprocess
import json
import time
import datetime
import pytz




def append_dict_to_jsonl(file_path, data_dict):
    print("1111111111111")
    with open(file_path, 'a', encoding='utf-8') as f:
        print("save the file_path to", file_path)
        json_str = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_str + '\n')


def get_free_gpus():
    try:
        # 执行 nvidia-smi 命令
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        # 分析输出结果
        free_gpus = []
        lines = output.strip().split("\n")
        for line in lines:
            index, memory_used = line.split(", ")
            if int(memory_used) <= 100:
                free_gpus.append(int(index))
        
        return free_gpus
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_start_time():
    start_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(start_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_start_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_start_time


def get_end_time():
    end_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(end_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_end_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_end_time
