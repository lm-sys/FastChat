## Chatbot Arena

Currently, Chatbot Arena is powered by FastChat. Here is how you can launch an instance of Chatbot Arena locally.

Create a file `api_endpoint.json` and record the the api endpoints of the models you want to serve, for example:
```
{
    "gpt-4o-2024-05-13": {
        "model_name": "gpt-4o-2024-05-13",
        "api_base": "https://api.openai.com/v1",
        "api_type": "openai",
        "api_key": [Insert API Key],
        "anony_only": false
    }
}
```

If you want to serve your own model using local gpus, following the instructions in [Section: Serving with Web GUI](../../README.md).

To launch a gradio web server, run `gradio_web_server_multi.py`.
```
python gradio_web_server_multi.py --port 8080 --share --register-api-endpoint-file api_endpoint.json
```