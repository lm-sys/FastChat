# Chatbot Arena
Chatbot Arena is an LLM benchmark platform featuring anonymous, randomized battles, available at https://chat.lmsys.org.
We invite the entire community to join this benchmarking effort by contributing your votes and models.

## How to add a new model
If you want to see a specific model in the arena, you can follow the methods below.

### Method 1: Hosted by 3rd party API providers or yourself
If you have a model hosted by a 3rd party API provider or yourself, please give us the access to an API endpoint.
  - We prefer OpenAI-compatible APIs, so we can reuse our [code](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/api_provider.py) for calling OpenAI models.
  - If you have your own API protocol, please follow the [instructions](model_support.md) to add them. Contribute your code by sending a pull request.

### Method 2: Hosted by LMSYS
1. Contribute the code to support this model in FastChat by submitting a pull request. See [instructions](model_support.md).
2. After the model is supported, we will try to schedule some compute resources to host the model in the arena. However, due to the limited resources we have, we may not be able to serve every model. We will select the models based on popularity, quality, diversity, and other factors.


## How to launch vision arena

1. Run `python3 -m fastchat.serve.controller` to start the controller and begin registering local model workers and API-provided workers.
2. Run `python3 -m fastchat.serve.sglang_worker --model-path <model-path> --tokenizer-path <tokenizer-path>` to run local vision-language models. Currently supported models include the LLaVA and Yi-VL series.
3. If you are using a 3rd party model with an API provider (e.g. GPT-4-V, Gemini 1.5), please follow the instructions [model_support.md](model_support.md) to add a json file `api_endpoints.json`.
4. Run the gradio server with the `--vision-arena` flag on.
5. To run and store images into a remote directory, add the flag: `--use-remote-storage`
6. To run and allow samples of random questions, add `--random_questions metadata_sampled.json`. Check sections below for how to generate this.

Example command:
```
python3 -m fastchat.serve.gradio_web_server_multi --share --register-api-endpoint-file api_endpoints.json --vision-arena --use-remote-storage --random-questions metadata_sampled.json
```

### NSFW and CSAM Detection
1. Adding NSFW Endpoint and API key: Please add the following environment variables to run the NSFW moderation filter for images: 
  - `AZURE_IMG_MODERATION_ENDPOINT`: This is the endpoint that the NSFW moderator is hosted (e.g. https://{endpoint}/contentmoderator/moderate/v1.0/ProcessImage/Evaluate). Change the `endpoint` to your own.
  - `AZURE_IMG_MODERATION_API_KEY`: Your API key to run this endpoint.
2. Adding CSAM API key:
  - `PHOTODNA_API_KEY`: The API key that runs the CSAM detector endpoint.

Example in `~/.bashrc`:
```
export AZURE_IMG_MODERATION_ENDPOINT=https://<endpoint>/contentmoderator/moderate/v1.0/ProcessImage/Evaluate
export AZURE_IMG_MODERATION_API_KEY=<api-key>
export PHOTODNA_API_KEY=<api-key>
```

### Adding Random Samples for VQA
We provide random samples of example images for users to interact with coming from various datasets including DocVQA, RealWorldQA, ChartQA and VizWiz-VQA.
1. Download the images and generate random questions file by running `python fastchat/serve/vision/create_vqa_examples_dir.py`