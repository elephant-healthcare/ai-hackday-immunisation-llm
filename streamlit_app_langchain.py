import os
import streamlit as st

# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from rag_langchain import create_conversational_chain

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")

embedding_model: str = "text-embedding-3-small"

@st.cache_resource(ttl="1h")
def create_retriever():
    from rag_langchain import create_retriever
    return create_retriever()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Setup memory for contextual conversation
message_history = StreamlitChatMessageHistory()
qa_chain = create_conversational_chain(create_retriever(), message_history)


if len(message_history.messages) == 0 or st.sidebar.button("Clear message history"):
    message_history.clear()
    message_history.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in message_history.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        #response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        response = qa_chain.invoke(
            {"question": user_query}, 
            config={ "callbacks": [retrieval_handler, stream_handler]}
        )
