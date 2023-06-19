# python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --model-names gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002
# export OPENAI_API_BASE=http://localhost:8000/v1
# export OPENAI_API_KEY=EMPTY
# wget https://raw.githubusercontent.com/hwchase17/langchain/v0.0.200/docs/modules/state_of_the_union.txt

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator


def test_chain():
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


if __name__ == "__main__":
    test_chain()
