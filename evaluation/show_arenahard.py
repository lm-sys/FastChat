import json

# show general result

dataset_model_names = ["gpt-4o-mini", "gpt-4o", "llama31-70b", "llama31-405b", "gemini-v1.5-pro", "claude-3.5-sonnet"]

# dataset_model_names = ["claude-3.5-sonnet"]

for dataset_model_name in dataset_model_names:

    fname = "eval_results/arena_hard_{}.jsonl".format(dataset_model_name)

    with open(fname, "r") as f:
        lines = f.readlines()

    results = [json.loads(line) for line in lines]
    results = sorted(results, key=lambda x: x["question_id"])

    # count "grade": "base" or "grade": "agent"
    base_count = 0
    agent_count = 0
    total = len(results)
    for result in results:
        if result["grade"] == "base":
            base_count += 1
        elif result["grade"] == "agent":
            agent_count += 1

    # break down
    agent_sample_files = "eval_samples/arena_hard_50_{}-agent.jsonl".format(dataset_model_name)
    with open(agent_sample_files, "r") as f:
        lines = f.readlines()
    agent_results = [json.loads(line) for line in lines]
    agent_results = sorted(agent_results, key=lambda x: x["question_id"])

    base_sample_files = "eval_samples/arena_hard_50_{}.jsonl".format(dataset_model_name)
    with open(base_sample_files, "r") as f:
        lines = f.readlines()
    base_results = [json.loads(line) for line in lines]
    base_results = sorted(base_results, key=lambda x: x["question_id"])
    

    # count "search_done": True
    search_done_count = 0
    non_search_done_count = 0
    search_agent_better = 0
    non_search_agent_better = 0
    for i in range(len(results)):
        agent_result = agent_results[i]
        base_result = base_results[i]
        llm_as_judge_result = results[i]
        if agent_result["search_done"]:
            search_done_count += 1
            if llm_as_judge_result["grade"] == "agent":
                search_agent_better += 1
        if agent_result["search_done"] == False:
            # print("Agent Result =====================")
            # print(agent_result["response"])
            # print("Base Result =====================")
            # print(base_result["response"])
            # print("LLM as Judge Result =====================")
            # print(llm_as_judge_result["grade"])
            non_search_done_count += 1
            if llm_as_judge_result["grade"] == "agent":
                non_search_agent_better += 1
                # print("-" * 50)
                # print("Question ID: {}".format(agent_result["question_id"]))
                # print(agent_result["response"])
                # if "NETWORK ERROR" in agent_result["response"]:
                #     print("Question ID: {}".format(agent_result["question_id"]))
            # exit()
    print("-" * 50)
    # print("Model: {}".format(dataset_model_name))
    # print("Search Done: {} ({}/{})".format(search_done_count/total, search_done_count, total))
    # print("  - Agent")
    # print("    - No search Acc: {} ({}/{})".format(non_search_agent_better/non_search_done_count, non_search_agent_better, non_search_done_count))
    # print("    - Search Acc: {} ({}/{})".format(search_agent_better/search_done_count, search_agent_better, search_done_count))
    # print("    - Overall: {} ({}/{})".format((search_agent_better+non_search_agent_better)/total, search_agent_better+non_search_agent_better, total))
    # print("  - Base")
    # print("    - No search Acc: {} ({}/{})".format(1-non_search_agent_better/non_search_done_count, non_search_done_count-non_search_agent_better, non_search_done_count))
    # print("    - Search Acc: {} ({}/{})".format(1-search_agent_better/search_done_count, search_done_count-search_agent_better, search_done_count))
    # print("    - Overall: {} ({}/{})".format(1-(search_agent_better+non_search_agent_better)/total, total-search_agent_better-non_search_agent_better, total))
    print(f"Model: {dataset_model_name}")
    print(f"Search Done: {search_done_count/total:.2%} ({search_done_count}/{total})")
    print("  - Agent")
    print(f"    - No search Acc: {non_search_agent_better/non_search_done_count:.2%} ({non_search_agent_better}/{non_search_done_count})")
    print(f"    - Search Acc: {search_agent_better/search_done_count:.2%} ({search_agent_better}/{search_done_count})")
    print(f"    - Overall: {(search_agent_better+non_search_agent_better)/total:.2%} ({search_agent_better+non_search_agent_better}/{total})")
    print("  - Base")
    print(f"    - No search Acc: {1-non_search_agent_better/non_search_done_count:.2%} ({non_search_done_count-non_search_agent_better}/{non_search_done_count})")
    print(f"    - Search Acc: {1-search_agent_better/search_done_count:.2%} ({search_done_count-search_agent_better}/{search_done_count})")
    print(f"    - Overall: {1-(search_agent_better+non_search_agent_better)/total:.2%} ({total-search_agent_better-non_search_agent_better}/{total})")

    # # show preference count
    # print(f"Model: {dataset_model_name}")
    # print(f"Base: {base_count/total:.4f}, Agent: {agent_count/total:.4f}")