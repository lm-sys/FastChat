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
  - name: openai_moderation
    list:
    - name: categories
      struct:
      - name: harassment
        dtype: bool
      - name: harassment/threatening
        dtype: bool
      - name: hate
        dtype: bool
      - name: hate/threatening
        dtype: bool
      - name: self-harm
        dtype: bool
      - name: self-harm/instructions
        dtype: bool
      - name: self-harm/intent
        dtype: bool
      - name: sexual
        dtype: bool
      - name: sexual/minors
        dtype: bool
      - name: violence
        dtype: bool
      - name: violence/graphic
        dtype: bool
    - name: category_scores
      struct:
      - name: harassment
        dtype: float64
      - name: harassment/threatening
        dtype: float64
      - name: hate
        dtype: float64
      - name: hate/threatening
        dtype: float64
      - name: self-harm
        dtype: float64
      - name: self-harm/instructions
        dtype: float64
      - name: self-harm/intent
        dtype: float64
      - name: sexual
        dtype: float64
      - name: sexual/minors
        dtype: float64
      - name: violence
        dtype: float64
      - name: violence/graphic
        dtype: float64
    - name: flagged
      dtype: bool
  splits:
  - name: train
    num_bytes: 2627534545
    num_examples: 1000000
  download_size: 1491379201
  dataset_size: 2627534545
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

---
## LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset

This dataset contains one million real-world conversations with 25 state-of-the-art LLMs.
It is collected from 210K unique IP addresses in the wild on the [Vicuna demo and Chatbot Arena website](https://chat.lmsys.org/) from April to August 2023.
Each sample includes a conversation ID, model name, conversation text in OpenAI API JSON format, detected language tag, and OpenAI moderation API tag.

To ensure the safe release of data, we have made our best efforts to remove all conversations that contain personally identifiable information (PII).
User consent is obtained through the "Terms of use" section on the data collection website.
In addition, we have included the OpenAI moderation API output for each message.
However, we have chosen to keep toxic conversations intact so that researchers can study the safety-related questions associated with LLM usage in real-world scenarios as well as the OpenAI moderation process.

**Basic Statistics**
| Key | Value |
| --- | --- |
| # Conversations | 1,000,000 |
| # Models | 25 |
| # Users | 210,479 |
| # Languages | 154 |
| Avg. # Turns per Sample | 2.0 |
| Avg. # Tokens per Prompt | 69.5 |
| Avg. # Tokens per Response | 214.5 |

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
