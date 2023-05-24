# Local LangChain with FastChat

LangChain is a framework for developing applications powered by language models. It provides a set of tools, components and interfaces that simplify the process of creating applications that are supported by large language models (LLMs) and chat models.FastChat's OpenAI-compatible API server enables to use LangChain with open models seamlessly.

## Launch RESTful API Server

There are some differences while launching API server for LangChain.

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Due to the fact that langchain checks whether the model's name belongs to OpenAI, we need to assign a faux OpenAI name to the Vicuna model. In essence, we're providing an OpenAI model name when loading the model.

```bash
python3 -m fastchat.serve.model_worker --model-name 'text-embedding-ada-002' --model-path /path/to/weights
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

## Try local LangChain

There are some examples you can try [here](../examples/langchain/qa.ipynb). You can also run your own code

~~~py
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
# wget https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt
loader = TextLoader('state_of_the_union.txt')
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])

llm = OpenAI(model="text-embedding-ada-002") # select your faux openai model name
# llm = OpenAI(model="gpt-3.5-turbo")


questions = [
             "who is the speaker", 
             "What did the president say about Ketanji Brown Jackson", 
             "What are the threats to America", 
             "Who are mentioned in the speech",
             "Who is the vice president",
             "How many projects were announced",
            ]

for query in questions:
    print("Query: ", query)
    print("Ans: ",index.query(query,llm=llm))
~~~