import os
import streamlit as st

from llama_index.llms.openai import OpenAI
import weave
import llama_index.core
from llama_index.core import set_global_handler

from query_llm import query_llm
from query_rag_llm import create_rag_llm
llm_name = "gpt-4o-mini"
model_temperature = 0.8

project = "ai_assistant_nurse_chat_app"
weave.init(project)
set_global_handler("wandb", run_args={"project": project})

@st.cache_resource
def rag_llm():
    return create_rag_llm()


st.title("🐘 AI Immunisation/MCH Nurse 🐘")
st.markdown(
    (
        "This demo allows you to query a LLM about immunisation and MCH topics."
    )
)

query_tab, rag_tab = st.tabs(["Query", "Knowledge Base Query"])

with query_tab:
    st.subheader("Query")
    st.markdown(
        (
            f"This is a simple wrapper around OpenAI's {llm_name} model."
        )
    )

    query_text = st.text_input("Ask a question:")
    if query_text:
        with st.spinner("Generating answer..."):
            response = query_llm(query_text)
        st.markdown(str(response))



with rag_tab:
    st.subheader("Knowledge Base Query")
    st.markdown(
        (
            f"This augments the {llm_name} model with a local knowledge base."
        )
    )

    st.session_state["rag_llm"] = rag_llm()

    query_text = st.text_input("Ask a question:", key="rag_query_text")
    if query_text:
        with st.spinner("Generating answer..."):
            response = st.session_state["rag_llm"].query(query_text)
        st.markdown(str(response))