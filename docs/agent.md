# Agent Arena Working Area

## The latest status

- Done:
  - [x] Complete the basic google search function in `fastchat/tools/search.py`. The pipeline now works for OpenAI search.
  - [x] Find some successful and failure cases using our naive search tool.
- TODOs:
  - [ ] Set an option to display the web search result or not (UI-related stuffs).
  - [ ] Scale the searching functions to other LLMs.
  - [ ] Run our pipeline on Arena Datasets to see if this naive search is sufficient.

- Note: Please run `./format.sh` before merging into the main branch.

**Note**: Please install packages and ensure you can successfully execute [Launch a WebUI with an API Model](https://github.com/tsunghan-wu/Agent_FastChat/blob/main/docs/model_support.md#api-based-models).

## Launch agent-enabled Chatbot Arena (for OpenAI APIs currently)

1. Specify the endpoint information in a JSON configuration file. For instance, create a file named `agent_api_endpoints.json`:

```
{
  "gpt4o": {
    "model_name": "gpt-4o-2024-08-06",
    "api_type": "openai",
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-******",
    "anony_only": false,
    "recommended_config": {
      "temperature": 0.7,
      "top_p": 1.0
    },
    "text-arena": true,
    "vision-arena": false,
    "agent-mode": true
  }
}
```

2. Launch the Gradio web server with the argument `--register api_endpoints.json`:

```bash
python3 -m fastchat.serve.gradio_web_server_agent --controller "" --share --register agent_api_endpoints.json
```

Now, you can open a browser and interact with the model.

## Examples:

1. Using Agents

```
User: What's the weather today?

GPT-4: 
{
    "thought": "The query asks about the current weather, which is dynamic and location-specific information. Since I don't have direct access to real-time weather data, I should use the available tool to search for the current weather conditions. The 'google_search' tool can help find this information by querying with relevant keywords.",
    "action": {
        "name": "google_search",
        "reason": "To obtain the current weather information from online sources.",
        "arguments": {
            "key_words": ["current weather"],
            "topk": 1
        }
    }
}

Then, we'll execute the google search function in the fastchat/tools/search.py where it currently only return a pre-defined string. You can also conduct follow-up (multi-round) chatting, it won't be a problem.
```

2. Without using agents

```
User: How are you?
GPT-4:
{
    "thought": "The query is asking for a status update or well-being check on myself as an assistant. This is a common conversational question and doesn't require additional information from external sources. I can answer this directly based on my designed functionality.",
    "answer": "I'm just a virtual assistant, so I don't have feelings or states of being, but I'm here and ready to help you with any questions or tasks you have!"
}
```

## Comparsing Responses Between Agent and Non-Agent Modes

You can use `compare_agents.ipynb` notebook to compare the response between standard LM and one augmented with our search ability
1. Start the server as usual
2. Run the notebook