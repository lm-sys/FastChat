"""Additional information of the models."""
from collections import namedtuple, OrderedDict
from typing import List


ModelInfo = namedtuple("ModelInfo", ["simple_name", "link", "description"])


model_info = OrderedDict()


def register_model_info(
    full_names: List[str], simple_name: str, link: str, description: str
):
    info = ModelInfo(simple_name, link, description)

    for full_name in full_names:
        model_info[full_name] = info


def get_model_info(name: str) -> ModelInfo:
    if name in model_info:
        return model_info[name]
    else:
        # To fix this, please use `register_model_info` to register your model
        return ModelInfo(
            name, "", "Register the description at fastchat/model/model_registry.py"
        )


register_model_info(
    [
        "IEITYuan/Yuan2-2B-Janus-hf",
        "IEITYuan/Yuan2-2B-hf",
        "IEITYuan/Yuan2-51B-hf",
        "IEITYuan/Yuan2-102B-hf",
    ],
    "IEIT-Yuan2",
    "https://github.com/IEIT-Yuan/Yuan-2.0",
    "Yuan2.0 is a new generation Fundamental Large Language Model developed by IEIT System.",
)

register_model_info(
    [
        "mixtral-8x7b-instruct-v0.1",
        "mistral-medium",
        "mistral-7b-instruct-v0.2",
        "mistral-7b-instruct",
    ],
    "Mixtral of experts",
    "https://mistral.ai/news/mixtral-of-experts/",
    "A Mixture-of-Experts model by Mistral AI",
)

register_model_info(
    [
        "qwen1.5-72b-chat",
        "qwen1.5-14b-chat",
        "qwen1.5-7b-chat",
        "qwen1.5-4b-chat",
        "qwen1.5-1.8b-chat",
        "qwen1.5-0.5b-chat",
        "qwen-14b-chat",
    ],
    "Qwen 1.5",
    "https://qwenlm.github.io/blog/qwen1.5/",
    "A large language model by Alibaba Cloud",
)

register_model_info(
    ["qwen-14b-chat"],
    "Qwen",
    "https://huggingface.co/Qwen",
    "A large language model by Alibaba Cloud",
)

register_model_info(
    ["bard-feb-2024", "bard-jan-24-gemini-pro"],
    "Bard",
    "https://bard.google.com/",
    "Bard by Google",
)

register_model_info(
    ["gemini-pro", "gemini-pro-dev-api"],
    "Gemini",
    "https://blog.google/technology/ai/google-gemini-pro-imagen-duet-ai-update/",
    "Gemini by Google",
)

register_model_info(
    ["deepseek-llm-67b-chat"],
    "DeepSeek LLM",
    "https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat",
    "An advanced language model by DeepSeek",
)

register_model_info(
    ["stripedhyena-nous-7b"],
    "StripedHyena-Nous",
    "https://huggingface.co/togethercomputer/StripedHyena-Nous-7B",
    "A chat model developed by Together Research and Nous Research.",
)

register_model_info(
    ["solar-10.7b-instruct-v1.0"],
    "SOLAR-10.7B-Instruct",
    "https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0",
    "A model trained using depth up-scaling by Upstage AI",
)

register_model_info(
    ["gpt-4-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview"],
    "GPT-4-Turbo",
    "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
    "GPT-4-Turbo by OpenAI",
)

register_model_info(
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0314",
        "gpt-3.5-turbo-0613",
    ],
    "GPT-3.5",
    "https://platform.openai.com/docs/models/gpt-3-5",
    "GPT-3.5-Turbo by OpenAI",
)

register_model_info(
    ["gpt-4", "gpt-4-0314", "gpt-4-0613"],
    "GPT-4",
    "https://openai.com/research/gpt-4",
    "GPT-4 by OpenAI",
)

register_model_info(
    ["claude-2.1", "claude-2.0"],
    "Claude",
    "https://www.anthropic.com/index/claude-2",
    "Claude 2 by Anthropic",
)

register_model_info(
    ["claude-1"],
    "Claude",
    "https://www.anthropic.com/index/introducing-claude",
    "Claude 1 by Anthropic",
)

register_model_info(
    ["claude-instant-1", "claude-instant-1.2"],
    "Claude Instant",
    "https://www.anthropic.com/index/introducing-claude",
    "Claude Instant by Anthropic",
)

register_model_info(
    ["nous-hermes-2-mixtral-8x7b-dpo"],
    "Nous-Hermes-2-Mixtral-8x7B-DPO",
    "https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Nous Hermes finetuned from Mixtral 8x7B",
)

register_model_info(
    ["openchat-3.5-0106", "openchat-3.5"],
    "OpenChat 3.5",
    "https://github.com/imoneoi/openchat",
    "An open model fine-tuned on Mistral-7B using C-RLFT",
)

register_model_info(
    ["deepseek-llm-67b-chat"],
    "DeepSeek LLM",
    "https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat",
    "An advanced language model by DeepSeek",
)

register_model_info(
    ["stripedhyena-nous-7b"],
    "StripedHyena-Nous",
    "https://huggingface.co/togethercomputer/StripedHyena-Nous-7B",
    "A chat model developed by Together Research and Nous Research.",
)

register_model_info(
    ["llama2-70b-steerlm-chat"],
    "Llama2-70B-SteerLM-Chat",
    "https://huggingface.co/nvidia/Llama2-70B-SteerLM-Chat",
    "A Llama fine-tuned with SteerLM method by NVIDIA",
)

register_model_info(
    ["pplx-70b-online", "pplx-7b-online"],
    "pplx-online-llms",
    "https://blog.perplexity.ai/blog/introducing-pplx-online-llms",
    "Online LLM API by Perplexity AI",
)

register_model_info(
    ["openhermes-2.5-mistral-7b"],
    "OpenHermes-2.5-Mistral-7B",
    "https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B",
    "A mistral-based model fine-tuned on 1M GPT-4 outputs",
)

register_model_info(
    ["starling-lm-7b-alpha"],
    "Starling-LM-7B-alpha",
    "https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha",
    "An open model trained using RLAIF by Berkeley",
)

register_model_info(
    ["tulu-2-dpo-70b"],
    "Tulu 2",
    "https://huggingface.co/allenai/tulu-2-dpo-70b",
    "An instruction and RLHF model by UW/AllenAI",
)

register_model_info(
    ["yi-34b-chat", "yi-6b-chat"],
    "Yi-Chat",
    "https://huggingface.co/01-ai/Yi-34B-Chat",
    "A large language model by 01 AI",
)

register_model_info(
    ["llama-2-70b-chat", "llama-2-34b-chat", "llama-2-13b-chat", "llama-2-7b-chat"],
    "Llama 2",
    "https://ai.meta.com/llama/",
    "Open foundation and fine-tuned chat models by Meta",
)

register_model_info(
    [
        "vicuna-33b",
        "vicuna-33b-v1.3",
        "vicuna-13b",
        "vicuna-13b-v1.5",
        "vicuna-7b",
        "vicuna-7b-v1.5",
    ],
    "Vicuna",
    "https://lmsys.org/blog/2023-03-30-vicuna/",
    "A chat assistant fine-tuned on user-shared conversations by LMSYS",
)

register_model_info(
    ["chatglm3-6b", "chatglm2-6b", "chatglm-6b"],
    "ChatGLM",
    "https://chatglm.cn/blog",
    "An open bilingual dialogue language model by Tsinghua University",
)

register_model_info(
    ["tenyxchat-7b-v1"],
    "TenyxChat-7B",
    "https://huggingface.co/tenyx/TenyxChat-7B-v1",
    "An open model DPO trained on top of OpenChat-3.5 using Tenyx fine-tuning",
)

register_model_info(
    ["zephyr-7b-beta", "zephyr-7b-alpha"],
    "Zephyr",
    "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha",
    "A chatbot fine-tuned from Mistral by Hugging Face",
)

register_model_info(
    ["notus-7b-v1"],
    "Notus",
    "https://huggingface.co/argilla/notus-7b-v1",
    "A chatbot fine-tuned from Zephyr SFT by Argilla",
)

register_model_info(
    ["catppt"],
    "CatPPT",
    "https://huggingface.co/rishiraj/CatPPT",
    "A chatbot fine-tuned from a SLERP merged model by Rishiraj Acharya",
)

register_model_info(
    ["TinyLlama"],
    "TinyLlama",
    "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.",
)

register_model_info(
    [
        "codellama-70b-instruct",
        "codellama-34b-instruct",
        "codellama-13b-instruct",
        "codellama-7b-instruct",
    ],
    "Code Llama",
    "https://ai.meta.com/blog/code-llama-large-language-model-coding/",
    "Open foundation models for code by Meta",
)

register_model_info(
    ["wizardlm-70b", "wizardlm-30b", "wizardlm-13b"],
    "WizardLM",
    "https://github.com/nlpxucan/WizardLM",
    "An instruction-following LLM using evol-instruct by Microsoft",
)

register_model_info(
    ["wizardcoder-15b-v1.0"],
    "WizardLM",
    "https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder",
    "Empowering Code Large Language Models with Evol-Instruct",
)

register_model_info(
    ["mpt-7b-chat", "mpt-30b-chat"],
    "MPT-Chat",
    "https://www.mosaicml.com/blog/mpt-30b",
    "A chatbot fine-tuned from MPT by MosaicML",
)

register_model_info(
    ["guanaco-33b", "guanaco-65b"],
    "Guanaco",
    "https://github.com/artidoro/qlora",
    "A model fine-tuned with QLoRA by UW",
)

register_model_info(
    ["gpt4all-13b-snoozy"],
    "GPT4All-Snoozy",
    "https://github.com/nomic-ai/gpt4all",
    "A finetuned LLaMA model on assistant style data by Nomic AI",
)

register_model_info(
    ["koala-13b"],
    "Koala",
    "https://bair.berkeley.edu/blog/2023/04/03/koala",
    "A dialogue model for academic research by BAIR",
)

register_model_info(
    ["RWKV-4-Raven-14B"],
    "RWKV-4-Raven",
    "https://huggingface.co/BlinkDL/rwkv-4-raven",
    "An RNN with transformer-level LLM performance",
)

register_model_info(
    ["alpaca-13b"],
    "Alpaca",
    "https://crfm.stanford.edu/2023/03/13/alpaca.html",
    "A model fine-tuned from LLaMA on instruction-following demonstrations by Stanford",
)

register_model_info(
    ["oasst-pythia-12b"],
    "OpenAssistant (oasst)",
    "https://open-assistant.io",
    "An Open Assistant for everyone by LAION",
)

register_model_info(
    ["oasst-sft-7-llama-30b"],
    "OpenAssistant (oasst)",
    "https://open-assistant.io",
    "An Open Assistant for everyone by LAION",
)

register_model_info(
    ["palm-2"],
    "PaLM 2 Chat",
    "https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023",
    "PaLM 2 for Chat (chat-bison@001) by Google",
)

register_model_info(
    ["llama-7b", "llama-13b"],
    "LLaMA",
    "https://arxiv.org/abs/2302.13971",
    "Open and efficient foundation language models by Meta",
)

register_model_info(
    ["open-llama-7b-v2-open-instruct", "open-llama-7b-open-instruct"],
    "Open LLaMa (Open Instruct)",
    "https://medium.com/vmware-data-ml-blog/starter-llm-for-the-enterprise-instruction-tuning-openllama-7b-d05fc3bbaccc",
    "Open LLaMa fine-tuned on instruction-following data by VMware",
)

register_model_info(
    ["dolly-v2-12b"],
    "Dolly",
    "https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm",
    "An instruction-tuned open large language model by Databricks",
)

register_model_info(
    ["stablelm-tuned-alpha-7b"],
    "StableLM",
    "https://github.com/stability-AI/stableLM",
    "Stability AI language models",
)

register_model_info(
    ["codet5p-6b"],
    "CodeT5p-6b",
    "https://huggingface.co/Salesforce/codet5p-6b",
    "Code completion model released by Salesforce",
)

register_model_info(
    ["fastchat-t5-3b", "fastchat-t5-3b-v1.0"],
    "FastChat-T5",
    "https://huggingface.co/lmsys/fastchat-t5-3b-v1.0",
    "A chat assistant fine-tuned from FLAN-T5 by LMSYS",
)

register_model_info(
    ["phoenix-inst-chat-7b"],
    "Phoenix-7B",
    "https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b",
    "A multilingual chat assistant fine-tuned from Bloomz to democratize ChatGPT across languages by CUHK(SZ)",
)

register_model_info(
    ["realm-7b-v1"],
    "ReaLM",
    "https://github.com/FreedomIntelligence/ReaLM",
    "A chatbot fine-tuned from LLaMA2 with data generated via iterative calls to UserGPT and ChatGPT by CUHK(SZ) and SRIBD.",
)

register_model_info(
    ["billa-7b-sft"],
    "BiLLa-7B-SFT",
    "https://huggingface.co/Neutralzz/BiLLa-7B-SFT",
    "An instruction-tuned bilingual LLaMA with enhanced reasoning ability by an independent researcher",
)

register_model_info(
    ["h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2"],
    "h2oGPT-GM-7b",
    "https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2",
    "An instruction-tuned OpenLLaMA with enhanced conversational ability by H2O.ai",
)

register_model_info(
    ["baize-v2-7b", "baize-v2-13b"],
    "Baize v2",
    "https://github.com/project-baize/baize-chatbot#v2",
    "A chatbot fine-tuned from LLaMA with ChatGPT self-chat data and Self-Disillation with Feedback (SDF) by UCSD and SYSU.",
)

register_model_info(
    [
        "airoboros-l2-7b-2.1",
        "airoboros-l2-13b-2.1",
        "airoboros-c34b-2.1",
        "airoboros-l2-70b-2.1",
    ],
    "airoboros",
    "https://huggingface.co/jondurbin/airoboros-l2-70b-2.1",
    "An instruction-tuned LlaMa model tuned with 100% synthetic instruction-response pairs from GPT4",
)

register_model_info(
    [
        "spicyboros-7b-2.2",
        "spicyboros-13b-2.2",
        "spicyboros-70b-2.2",
    ],
    "spicyboros",
    "https://huggingface.co/jondurbin/spicyboros-70b-2.2",
    "De-aligned versions of the airoboros models",
)

register_model_info(
    ["Robin-7b-v2", "Robin-13b-v2", "Robin-33b-v2"],
    "Robin-v2",
    "https://huggingface.co/OptimalScale/robin-7b-v2-delta",
    "A chatbot fine-tuned from LLaMA-7b, achieving competitive performance on chitchat, commonsense reasoning and instruction-following tasks, by OptimalScale, HKUST.",
)

register_model_info(
    ["manticore-13b-chat"],
    "Manticore 13B Chat",
    "https://huggingface.co/openaccess-ai-collective/manticore-13b-chat-pyg",
    "A chatbot fine-tuned from LlaMa across several CoT and chat datasets.",
)

register_model_info(
    ["redpajama-incite-7b-chat"],
    "RedPajama-INCITE-7B-Chat",
    "https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat",
    "A chatbot fine-tuned from RedPajama-INCITE-7B-Base by Together",
)

register_model_info(
    [
        "falcon-7b",
        "falcon-7b-instruct",
        "falcon-40b",
        "falcon-40b-instruct",
        "falcon-180b",
        "falcon-180b-chat",
    ],
    "Falcon",
    "https://huggingface.co/tiiuae/falcon-180B",
    "TII's flagship series of large language models",
)

register_model_info(
    ["tigerbot-7b-sft"],
    "Tigerbot",
    "https://huggingface.co/TigerResearch/tigerbot-7b-sft",
    "A large-scale language model (LLM) with multiple languages and tasks.",
)

register_model_info(
    ["internlm-chat-7b", "internlm-chat-7b-8k"],
    "InternLM",
    "https://huggingface.co/internlm/internlm-chat-7b",
    "A multi-language large-scale language model (LLM), developed by SHLAB.",
)

register_model_info(
    ["Qwen-7B-Chat"],
    "Qwen",
    "https://huggingface.co/Qwen/Qwen-7B-Chat",
    "A multi-language large-scale language model (LLM), developed by Damo Academy.",
)

register_model_info(
    ["Llama2-Chinese-13b-Chat", "LLama2-Chinese-13B"],
    "Llama2-Chinese",
    "https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat",
    "A multi-language large-scale language model (LLM), developed by FlagAlpha.",
)

register_model_info(
    ["Chinese-Alpaca-2-7B", "Chinese-Alpaca-2-13B"],
    "Chinese-Alpaca",
    "https://huggingface.co/hfl/chinese-alpaca-2-13b",
    "New extended Chinese vocabulary beyond Llama-2, open-sourcing the Chinese LLaMA-2 and Alpaca-2 LLMs.",
)

register_model_info(
    ["Vigogne-2-7B-Instruct", "Vigogne-2-13B-Instruct"],
    "Vigogne-Instruct",
    "https://huggingface.co/bofenghuang/vigogne-2-7b-instruct",
    "A French large language model (LLM) optimized for instruction-following, developed by Bofeng Huang",
)

register_model_info(
    ["Vigogne-2-7B-Chat", "Vigogne-2-13B-Chat"],
    "Vigogne-Chat",
    "https://huggingface.co/bofenghuang/vigogne-2-7b-chat",
    "A French large language model (LLM) optimized for instruction-following and multi-turn dialogues, developed by Bofeng Huang",
)

register_model_info(
    ["stable-vicuna-13B-HF"],
    "stable-vicuna",
    "https://huggingface.co/TheBloke/stable-vicuna-13B-HF",
    "A Vicuna model fine-tuned using RLHF via PPO on various conversational and instructional datasets.",
)

register_model_info(
    ["deluxe-chat-v1", "deluxe-chat-v1.1", "deluxe-chat-v1.2"],
    "DeluxeChat",
    "",
    "Deluxe Chat",
)

register_model_info(
    [
        "Xwin-LM-7B-V0.1",
        "Xwin-LM-13B-V0.1",
        "Xwin-LM-70B-V0.1",
        "Xwin-LM-7B-V0.2",
        "Xwin-LM-13B-V0.2",
    ],
    "Xwin-LM",
    "https://github.com/Xwin-LM/Xwin-LM",
    "Chat models developed by Xwin-LM team",
)

register_model_info(
    ["lemur-70b-chat"],
    "Lemur-Chat",
    "https://huggingface.co/OpenLemur/lemur-70b-chat-v1",
    "An openly accessible language model optimized for both natural language and coding capabilities ",
)

register_model_info(
    ["Mistral-7B-OpenOrca"],
    "Open-Orca",
    "https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca",
    "A fine-tune of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) using [OpenOrca dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)",
)

register_model_info(
    ["dolphin-2.2.1-mistral-7b"],
    "dolphin-mistral",
    "https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b",
    "An uncensored fine-tuned Mistral 7B",
)

register_model_info(
    [
        "AquilaChat-7B",
        "AquilaChat2-7B",
        "AquilaChat2-34B",
    ],
    "Aquila-Chat",
    "https://huggingface.co/BAAI/AquilaChat2-34B",
    "Chat models developed by BAAI team",
)

register_model_info(
    ["xDAN-L1-Chat-RL-v1"],
    "xDAN-L1-Chat",
    "https://huggingface.co/xDAN-AI/xDAN-L1-Chat-RL-v1",
    "A large language chat model created by xDAN-AI.",
)

register_model_info(
    ["MetaMath-70B-V1.0", "MetaMath-7B-V1.0"],
    "MetaMath",
    "https://huggingface.co/meta-math",
    "A finetune of Llama2 on [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) that specializes in mathematical reasoning.",
)

register_model_info(
    ["Yuan2-2B-hf", "Yuan2-51B-hf", "Yuan2-102B-hf"],
    "IEIYuan",
    "https://huggingface.co/IEITYuan",
    "A Basemodel developed by IEI.",
)

register_model_info(
    [
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "llava-v1.6-vicuna-7b",
        "llava-v1.6-mistral-7b",
        "llava-v1.5-13b",
        "llava-v1.5-7b",
    ],
    "LLaVA",
    "https://github.com/haotian-liu/LLaVA",
    "an open large language and vision assistant",
)

register_model_info(
    ["gemma-7b-it", "gemma-2b-it"],
    "Gemma",
    "https://blog.google/technology/developers/gemma-open-models/",
    "Gemma by Google",
)

register_model_info(
    [
        "cllm/consistency-llm-7b-codesearchnet",
        "cllm/consistency-llm-7b-gsm8k",
        "cllm/consistency-llm-7b-sharegpt48k",
        "cllm/consistency-llm-7b-spider",
    ],
    "consistency-llm",
    "https://huggingface.co/cllm",
    "consistency-llm is a new generation of parallel decoder LLMs with fast generation speed.",
)