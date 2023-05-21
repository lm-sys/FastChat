# export OPENAI_API_BASE=http://localhost:8000/v1
# export OPENAI_API_KEY=EMPTY

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

template = """{history}
Human: {human_input}
Assistant:"""

def test_embedding():
    embeddings = OpenAIEmbeddings()
    texts = ["Why does the chicken cross the road", "To be honest", "Long time ago"]
    query_result = embeddings.embed_query(texts[0])
    doc_result = embeddings.embed_documents(texts)
    assert np.allclose(query_result, doc_result[0], atol=1e-3)

def test_chain():

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    chain = LLMChain(
        llm=OpenAI(model="text-embedding-ada-002", temperature=1), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(k=2),
    )
    output = chain.predict(human_input="ls ~")
    print(output)

if __name__ == "__main__":
    test_embedding()
    test_chain()

