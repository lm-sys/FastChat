import json

def jsonl_to_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data_jsonl = file.readlines()

    data_dict = {}
    counter = 1

    for item in data_jsonl:
        item_json = json.loads(item)

        topic = item_json.get("policy")
        question = item_json.get("question")

        if topic not in data_dict:
            data_dict[topic] = {"topic_id": len(data_dict) + 1, "topic": topic, "questions": {}}

        data_dict[topic]["questions"][str(counter)] = question

        counter +=1

    output_list = list(data_dict.values())

    return output_list

def save_as_json(output_list, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(output_list, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    output_list = jsonl_to_json('/ML-A100/home/tianyu/vllm_infer/vllm/choice_100_v2.jsonl') # 请替换成实际路径
    save_as_json(output_list, '/ML-A100/home/tianyu/vllm_infer/vllm/data_choice_sample100/questions.json') # 请替换成实际路径

