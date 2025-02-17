import os
import streamlit as st

from llama_index.llms.openai import OpenAI
import weave

from query_llm import query_llm
from query_rag_llm import create_rag_llm, create_query_rag_llm_v2
from ragas_evaluation import generate_eval_df

project = "ai_assistant_nurse_chat_app"
weave.init(project)

@st.cache_resource
def rag_llm():
    return create_rag_llm()

@st.cache_resource
def create_query_rag_with_ic_llm():
    return create_query_rag_llm_v2()


st.title("🐘 AI Immunisation/MCH Nurse 🐘")
st.markdown(
    (
        "This demo allows you to query a LLM about immunisation and MCH topics."
    )
)

final_tab, rag_tab, query_tab, eval_tab = st.tabs(
    ["Final Model", "Knowledge Base augmented", "Basic ChatGpt wrapper", "Evaluation"],
    )

with query_tab:
    llm_name = "gpt-4o-mini"
    model_temperature = 0.8

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
            f"This augments the LLM with a clinical guidelines knowledge base."
        )
    )

    st.session_state["rag_llm"] = rag_llm()

    query_text = st.text_input("Ask a question:", key="rag_query_text")
    if query_text:
        with st.spinner("Generating answer..."):
            response = st.session_state["rag_llm"].query(query_text)
        st.markdown(str(response))

with final_tab:
    st.subheader("KB + Intent Classifier model")
    st.markdown(
        (
            f"This augments the LMM with a clinical guidelines knowledge base and an intent classifier."
        )
    )

    st.session_state["query_rag_with_ic_llm"] = create_query_rag_with_ic_llm()

    query_text = st.text_input("Ask a question:", key="final_query_text")
    if query_text:
        response = None
        with st.spinner("Generating answer..."):
            response = st.session_state["query_rag_with_ic_llm"](query_text)
            print(response.__dict__.keys())
        st.markdown(str(response))

        container = st.container()
        container.status = st.status("**Retrieved Guidelines**")
        for idx, node in enumerate(response.source_nodes):
            container.status.write(f"**Document {idx} from {node.metadata['file_name']}**")
            container.status.markdown(node.text)
        container.status.update(state="complete")

with eval_tab:
    st.subheader("Test cases evaluation")
    #eval_df = generate_eval_df(query_engine=st.session_state["rag_llm"])
    #st.dataframe(eval_df)
