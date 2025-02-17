# Replacing ConversationalRetrievalChain as deprecated
# https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html
# https://python.langchain.com/docs/how_to/chatbots_retrieval/
    # https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html

import os
from langchain.chains import (
    create_retrieval_chain,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

embedding_model = "text-embedding-3-small"
model_name = "gpt-4o"
temperature = 0.1


SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

SYSTEM_TEMPLATE = """<guidelines>{context}</guidelines>
    You are a helpful AI nurse conciselyanswering health questions based on the guidelines above.
    If question cannot be clearly answered from the guidelines,
    answer with 'I'm sorry, I can't answer that question based on my current clinical knowledge.'"""

QUESTION_ANSWERING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        ("human", "<question>{input}</question>"),
    ]
)

DOCS_DIR = "./docs"

def create_retriever(chunk_size=1500, chunk_overlap=200):
    docs = DirectoryLoader(DOCS_DIR, glob="**/*.md").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    vectordb = InMemoryVectorStore.from_documents(
        splits,
        OpenAIEmbeddings(model=embedding_model)
    )

    return vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 1})

def create_chain():

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=temperature,
        streaming=True
    )
    retriever = create_retriever()
    combine_docs_chain = create_stuff_documents_chain(
        llm, QUESTION_ANSWERING_PROMPT
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def create_conversational_chain(retriever, message_history):
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)

    # Setup LLM and QA chain
    model_name = "gpt-4o"
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0, streaming=True
    )
    # https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.3.18/libs/langchain/langchain/chains/conversational_retrieval/base.py
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    return qa_chain

if __name__ == "__main__":
    from langchain.callbacks.base import BaseCallbackHandler

    class Handler(BaseCallbackHandler):
        def on_retriever_start(self, serialized: dict, query: str, **kwargs):
            print(f"**query:** {query}")

        def on_retriever_end(self, documents, **kwargs):
            for idx, doc in enumerate(documents):
                print(idx, doc.metadata["source"])

    chain = create_chain()
    result = chain.invoke({"input": "What vaccines should my 3 month old baby get?"}, config={"callbacks": [Handler()]})
    print(result)