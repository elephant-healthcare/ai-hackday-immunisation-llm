import os
import streamlit as st

from llama_index.llms.openai import OpenAI
import weave

from query_llm import query_llm
from query_rag_llm import create_rag_llm
from ragas_evaluation import generate_eval_df
llm_name = "gpt-4o-mini"
model_temperature = 0.8

project = "assistant_nurse_clinical_feedback"
weave.init(project)

@st.cache_resource
def rag_llm():
    return create_rag_llm()


st.title("🐘 AI Immunisation/MCH Nurse Feedback 🐘")
st.markdown(
    (
        "This app allows you to query a LLM about immunisation and MCH topics. It is intended to be used by our clinical expert to provide feedback on the AI nurse's responses."
    )
)

rag_tab, query_tab, eval_tab = st.tabs(
    ["Knowledge Base augmented", "Basic ChatGpt wrapper"],
    )

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


with eval_tab:
    st.subheader("Test cases evaluation")
    eval_df = generate_eval_df(query_engine=st.session_state["rag_llm"])
    st.dataframe(eval_df)
