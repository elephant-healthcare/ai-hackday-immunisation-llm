import streamlit as st
import uuid

import weave

from query_llm import query_llm
from query_rag_llm import RagModel, create_rag_llm, create_query_rag_llm_v2


def init_states():
    """Set up session_state keys if they don't exist yet."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "calls" not in st.session_state:
        st.session_state["calls"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "rag_model" not in st.session_state:
        st.session_state["rag_model"] = RagModel(name="mistral-large-latest", chat_llm="mistral-large-latest")


# https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
def render_feedback_buttons(call_idx):
    """Renders thumbs up/down and text feedback for the call."""

    col1, col2, col3 = st.columns([1, 1, 4])

    print("render_feedback_buttons", len(st.session_state.calls), call_idx)

    with col1:
        if st.button("👍", key=f"thumbs_up_{call_idx}"):
            st.session_state.calls[call_idx].feedback.add_reaction("👍")

    with col2:
        if st.button("👎", key=f"thumbs_down_{call_idx}"):
            st.session_state.calls[call_idx].feedback.add_reaction("👎")

@st.cache_resource
def rag_llm():
    return create_rag_llm()

@st.cache_resource
def create_query_rag_with_ic_llm():
    return create_query_rag_llm_v2()


def display_old_messages():
    """Displays the conversation stored in st.session_state.messages with feedback buttons"""

    print("display_old_messages", st.session_state.messages)

    if len(st.session_state.messages) > 0:
        with st.chat_message(st.session_state.messages[-1]["role"]):
            st.markdown(st.session_state.messages[-1]["content"])
            render_feedback_buttons(len(st.session_state.calls) - 1)


def display_chat_input():
    if user_input := st.chat_input("Ask a question:"):
        # Immediately render new user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Save user message in session
        # st.session_state.messages.append({"role": "user", "content": prompt})

        # Attach Weave attributes for tracking of conversation instances
        with weave.attributes(
            {"session": st.session_state["session_id"], "env": "prod"}
        ):
            # Call the LLM
            rag_model = st.session_state["rag_model"]
            # self has to be passed as first argument as we are calling a decorated function
            # https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
            response, call = rag_model.predict.call(rag_model, query=user_input)
            st.session_state.calls.append(call)

            # Store the assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": response.response}
            )

            # Store the weave call object to link feedback to the specific response
            st.session_state.calls.append(call)


def display_user_input():
    query_text = st.text_input("Ask a question:", key="final_query_text")

    if query_text:
        response = None
        with st.spinner("Generating answer..."):
            rag_model = st.session_state["rag_model"]
            # self has to be passed as first argument as we are calling a decorated function
            # https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
            response, call = rag_model.predict.call(rag_model, query=query_text)
            st.session_state.calls.append(call)

        st.markdown(str(response))

        container = st.container()
        container.status = st.status("**Retrieved Guidelines**")
        for idx, node in enumerate(response.source_nodes):
            container.status.write(f"**Document {idx} from {node.metadata['file_name']}**")
            container.status.markdown(node.text)
        container.status.update(state="complete")
        
    # always attaching the last call
    render_feedback_buttons(call_idx=len(st.session_state.calls) - 1)


if __name__ == "__main__":
    project = "ai_assistant_nurse_chat_app"
    weave.init(project)

    init_states()

    st.title("🐘 AI Immunisation/MCH Nurse 🐘")
    st.markdown(
        (
            "This demo allows you to query a LLM about immunisation and MCH topics."
        )
    )

    final_tab, basic_tab = st.tabs(
        ["Final", "Basic ChatGpt wrapper"],
        )

    with basic_tab:

        st.subheader("Query")
        st.markdown(
            (
                f"Simple wrapper around ChatGPT4o."
            )
        )

        query_text = st.text_input("Ask a question:")
        if query_text:
            with st.spinner("Generating answer..."):
                response = query_llm(query_text)
            st.markdown(str(response))

    with final_tab:
        st.subheader("KB + Intent Classifier model")
        st.markdown(
            (
                f"This augments the LMM with a clinical guidelines knowledge base and an intent classifier."
            )
        )

        display_chat_input()
        display_old_messages()