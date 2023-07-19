# Local LangChain with FastChat

[LangChain](https://python.langchain.com/en/latest/index.html) is a library that facilitates the development of applications by leveraging large language models (LLMs) and enabling their composition with other sources of computation or knowledge.
FastChat's OpenAI-compatible [API server](openai_api.md) enables using LangChain with open models seamlessly.

## Launch RESTful API Server

Here are the steps to launch a local OpenAI API server for LangChain.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

LangChain uses OpenAI model names by default, so we need to assign some faux OpenAI model names to our local model.
Here, we use Vicuna as an example and use it for three endpoints: chat completion, completion, and embedding.
`--model-path` can be a local folder or a Hugging Face repo name.
See a full list of supported models [here](../README.md#supported-models).

```bash
python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path lmsys/vicuna-7b-v1.3
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

## Set OpenAI Environment

You can set your environment with the following commands.

Set OpenAI base url

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
```

Set OpenAI API key

```bash
export OPENAI_API_KEY=EMPTY
```

If you meet the following OOM error while creating embeddings, please set a smaller batch size by using environment variables.

~~~bash
openai.error.APIError: Invalid response object from API: '{"object":"error","message":"**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**\\n\\n(CUDA out of memory. Tried to allocate xxx MiB (GPU 0; xxx GiB total capacity; xxx GiB already allocated; xxx MiB free; xxx GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF)","code":50002}' (HTTP response code was 400)
~~~

You can try `export FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE=1`.

## Try local LangChain

Here is a question answerting example.

Download a text file.

```bash
wget https://raw.githubusercontent.com/hwchase17/langchain/v0.0.200/docs/modules/state_of_the_union.txt
```

Run LangChain.

~~~py
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("state_of_the_union.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="gpt-3.5-turbo")

questions = [
    "Who is the speaker",
    "What did the president say about Ketanji Brown Jackson",
    "What are the threats to America",
    "Who are mentioned in the speech",
    "Who is the vice president",
    "How many projects were announced",
]

for query in questions:
    print("Query:", query)
    print("Answer:", index.query(query, llm=llm))
~~~
