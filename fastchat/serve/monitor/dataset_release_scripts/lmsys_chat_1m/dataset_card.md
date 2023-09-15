---
dataset_info:
  features:
  - name: conversation_id
    dtype: string
  - name: model
    dtype: string
  - name: conversation
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: turn
    dtype: int64
  - name: language
    dtype: string
  splits:
  - name: train
    num_bytes: 2251577093
    num_examples: 1000000
  download_size: 1055034253
  dataset_size: 2251577093
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc
task_categories:
- conversational
language:
- en
- pt
- ru
- zh
- es
size_categories:
- 100K<n<1M
---
## Content

This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.
This dataset is collected from 237K unique IP addresses in the wild on the [Vicuna demo and Chatbot Arena website](https://chat.lmsys.org/).
The dataset is stored in JSON format, where each record includes a conversation ID, model name, language tag, and conversation history in OpenAI Chat API format.
We additionally provide the OpenAI moderation API tag for each message.

**Basic Statistics**
| Key | Value |
| --- | --- |
| # Conversations | 1,000,000 |
| # Models | 25 |
| # Users | 236,985 |
| # Languages | 155 |
| Avg. # Turns per Sample | 2.0 |
| Avg. # Tokens per Prompt | 69.5 |
| Avg. # Tokens per Response | 215.3 |

## Uniqueness and Potential Usage
This dataset features large-scale real-world conversations with LLMs.

We believe it will help the AI research community answer important questions around topics like:
- Characteristics and distributions of real-world user prompts
- AI safety and content moderation
- Training instruction-following models
- Improving and evaluating LLM evaluation methods
- Model selection and request dispatching algorithms

## Disclaimers and Terms
- This dataset includes offensive conversations. It is not intended for training dialogue agents without applying appropriate filtering measures. We are not responsible for any outputs of the models trained on this dataset.
- Statements or opinions made in this dataset do not reflect the views of researchers or institutions involved in the data collection effort.
- Users of this data are responsible for ensuring its appropriate use, which includes abiding by any applicable laws and regulations.
- Users of this data should adhere to the terms of use for a specific model when using its direct outputs.

## License
The user prompts are licensed under CC-BY-4.0, while the model outputs are licensed under CC-BY-NC-4.0.

## Citation
TODO
