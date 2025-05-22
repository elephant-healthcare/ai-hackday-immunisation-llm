import os
import uuid
import streamlit as st

from openai import OpenAI
# Using langfuse's openai wrapper for full tracing
from langfuse.openai import openai as langfuse_openai



from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

# to export as environment variables:
# export LANGFUSE_SECRET_KEY=sk-lf-4a436773-6d7a-4627-b0cd-76fb800a63e8
# export LANGFUSE_PUBLIC_KEY=pk-lf-503f4091-bca0-4b1f-bc1d-a9afa46645e6
# export LANGFUSE_HOST="https://cloud.langfuse.com"
langfuse = Langfuse()

PROMPT = langfuse.get_prompt("patient-comms-prompt", label="latest")

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if "generate_trace_ids" not in st.session_state:
        st.session_state["generate_trace_ids"] = []

def to_chat_ml_messages(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]

def generate_around_langfuse_wrapper(messages, prompt=PROMPT, model="gpt-4o-mini"):
    """
    Wrapper around langfuse_openai.chat.completions.create
    """
    langfuse_trace_id = str(uuid.uuid4())
    st.session_state["generate_trace_ids"].append(langfuse_trace_id)

    completion = langfuse_openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}] + messages,
            #stream=True, Seems problematic with langfuse tracing somehow
            session_id=st.session_state['session_id'],
            trace_id=langfuse_trace_id,
        )
    return completion.choices[0].message.content


# https://langfuse.com/docs/sdk/python/decorators
#Important: Make sure the as_type="generation" decorated function is called inside another @observe()-decorated function for it to have a top-level trace.
#@observe(as_type="generation")
@observe
def generate(messages, model="gpt-4o-mini"):
    session_id = st.session_state['session_id']
    trace_id = langfuse_context.get_current_trace_id()
    # Only once though
    st.session_state["generate_trace_ids"].append(trace_id)

    # Duplicated to ensure proper trace UI at both session/trace/observation levels
    langfuse_context.update_current_trace(
        session_id=session_id
    )
    langfuse_context.update_current_observation(
        input=messages, # ensures formatted in UI, see https://www.reddit.com/r/LangChain/comments/1jq14f4/langfuse_pretty_traces/
        prompt=PROMPT,
        model=model,
    )
    return nested_generate(messages, model)

# to link with prompt, see https://langfuse.com/docs/prompts/get-started
@observe(as_type="generation")
def nested_generate(messages, model="gpt-4o-mini"):
    session_id = st.session_state['session_id']
    trace_id = langfuse_context.get_current_trace_id()
    # Duplicated to ensure proper trace UI at both session/trace/observation levels
    langfuse_context.update_current_trace(
        session_id=session_id
    )
    langfuse_context.update_current_observation(
        input=messages, # ensures formatted in UI, see https://www.reddit.com/r/LangChain/comments/1jq14f4/langfuse_pretty_traces/
        prompt=PROMPT,
        model=model,
    )
    
    print("Generating completion with session_id", session_id, "and trace_id", trace_id)

    completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PROMPT.prompt}] + messages,
            #stream=True,
        )
    return completion.choices[0].message.content

def render_feedback_buttons(call_idx):
    """Renders thumbs up/down and text feedback for the call.
    Adapted from https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
    and https://langfuse.com/docs/scores/custom
    """

    thumb_up, thumb_down, _ = st.columns([1, 1, 4])
    trace_id = st.session_state["generate_trace_ids"][call_idx]
    print("render_feedback_buttons", len(st.session_state["generate_trace_ids"]), call_idx)

    with thumb_up:
        print("thumb_up", call_idx, trace_id)
        if st.button("👍", key=f"thumbs_up_{call_idx}"):            
            print("👍", call_idx, trace_id)
            langfuse.score(
                trace_id=trace_id,
                name="feedback",
                value="👍",
            )

    with thumb_down:
        if st.button("👎", key=f"thumbs_down_{call_idx}"):
            print("👎", call_idx, trace_id)
            langfuse.score(
                trace_id=trace_id,
                name="feedback",
                value="👎",
            )

st.title("Immunisation Communication Agent")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Patient messsage"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = generate(
            messages=to_chat_ml_messages(st.session_state.messages),
            model=st.session_state["openai_model"]
        )
        #response = st.write_stream(stream)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Unfortunately not firing when pressing buttons...
    #render_feedback_buttons(len(st.session_state["generate_trace_ids"]) - 1)
