# Chatbot Arena
Chatbot Arena is an LLM benchmark platform featuring anonymous, randomized battles, available at https://arena.lmsys.org.
We invite the entire community to join this benchmarking effort by contributing your votes and models.

## How to add a new model
If you want to see a specific model in the arena, you can follow the steps below.

1. Contribute code to support the new model in FastChat by submitting a pull request.  
   The goal is to make the following command work.
   ```
   python3 -m fastchat.serve.cli --model YOUR_MODEL_PATH
   ```

   You can run this example command to learn the code logic.
   ```
   python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0
   ```
   
   Some major files you need to modify include
   - Add a conversation template for your model at https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py, you can follow the existing templates and register the new one at `get_default_conv_template`.
   - Make sure functions like `load_model` and `generate_stream` work at https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py)
2. After the model is supported, we will try to schedule some computing resources to host the model in the arena.
   However, due to the limited resource we have, we may not be able to serve every model.
   We will select the models based on popularity, quality, diversity, and other factors.

## Roadmap
- [ ] Add ChatGPT into the arena.
- [ ] Implement better sampling algorithms to support a large number of models.
