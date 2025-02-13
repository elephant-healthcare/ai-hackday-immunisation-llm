import os
import streamlit as st

from llama_index.llms.openai import OpenAI

from constants import DEFAULT_TERM_STR, DEFAULT_TERMS, REFINE_TEMPLATE, TEXT_QA_TEMPLATE
from query_llm import query_llm

llm_name = "gpt-4o-mini"
model_temperature = 0.8

st.title("🐘 AI Immunisation/MCH Nurse 🐘")
st.markdown(
    (
        "This demo allows you to query a LLM about immunisation and MCH topics."
    )
)

query_tab, = st.tabs(["Query"])

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

