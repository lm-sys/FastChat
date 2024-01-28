import argparse
import json
import copy


def parse_json(path):
    results = []
    with open(path, "r") as file:
        for line in file:
            results.append(json.loads(line))
    
    print(len(results))
    return results


template = {"id": -1, 
            "model": "gpt3.5",
            "conversations": []
            }


def reformat(in_list, human_role, gpt_role):
    out = []
    for i, item in enumerate(in_list):
        assert human_role in item
        human = item[human_role]
        assert gpt_role in item
        gpt = item[gpt_role]
        out_item = copy.deepcopy(template)
        out_item["id"] = i
        out_item["conversations"] = []

        human_turn = {"from": "human",
                "value": human}
        gpt_turn = {"from": "gpt",
                "value": gpt}

        out_item["conversations"].append(human_turn)
        out_item["conversations"].append(gpt_turn)
        out.append(out_item)
    return out


def main(args):
	content = parse_json(args.in_file)
	print(f"In total {len(content)} files to processed...")
	out = reformat(content, args.human_role, args.gpt_role)
	print(f"In total {len(content)} files finished...")
	print(out[10])
	json.dump(out, open(args.out_file, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--in-file", type=str, required=True)
	parser.add_argument("--out-file", type=str, default="output.json")
	parser.add_argument("--human-role", type=str, default="problem")
	parser.add_argument("--gpt-role", type=str, default="solution")
	args = parser.parse_args()
	main(args)
