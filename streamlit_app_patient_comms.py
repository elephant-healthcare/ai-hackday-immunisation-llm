import datetime
import os
from typing import List, Optional
import uuid
import streamlit as st

from openai import OpenAI
# Using langfuse's openai wrapper for full tracing
from langfuse.openai import openai as langfuse_openai
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

# Structured output with Pydantic and OpenAI
# https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat
from pydantic import BaseModel
 
DEFAULT_OPENAI_MODEL = "o4-mini"

class ConversationTurn(BaseModel):
    guardian_next_chosen_due_date: Optional[datetime.date]
    child_vaccines: List[str]
    has_guardian_opted_out: bool
    has_guardian_deviated_from_intended_use: bool
    assistant_next_response: str

# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

# to export as environment variables:
# export LANGFUSE_SECRET_KEY=sk-lf-4a436773-6d7a-4627-b0cd-76fb800a63e8
# export LANGFUSE_PUBLIC_KEY=pk-lf-503f4091-bca0-4b1f-bc1d-a9afa46645e6
# export LANGFUSE_HOST="https://cloud.langfuse.com"
langfuse = Langfuse()

PROMPT = langfuse.get_prompt("patient-comms-prompt", label="latest")
print("PROMPT", PROMPT)

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if "generate_trace_ids" not in st.session_state:
        st.session_state["generate_trace_ids"] = []

def to_chat_ml_messages(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# https://langfuse.com/docs/sdk/python/decorators
# Important: Make sure the as_type="generation" decorated function is called inside another @observe()-decorated function for it to have a top-level trace.
@observe
def generate(messages, model):
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

@observe(as_type="generation")
def nested_generate(messages, model) -> ConversationTurn | None:
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

    completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": PROMPT.prompt + f"\n Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}"}] + messages,
            response_format=ConversationTurn,
        )
    
    parsed_response = completion.choices[0].message.parsed
    print("Parsed response:\n", parsed_response)
    return parsed_response


def display_structured_response(response: ConversationTurn):
    container = st.container()
    container.status = st.status("**Structured output**")
    container.status.markdown(f"```json\n{response.model_dump_json(indent=2) if response else response}\n```")
    container.status.update(state="complete")

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
    st.session_state["openai_model"] = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there, this is Elephant. We're doing a study with your local gvmt and want to ask about your child's vaccines. Reply YES to take part, or STOP to opt out."}
    ]

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
        st.markdown(response.assistant_next_response)
    st.session_state.messages.append({"role": "assistant", "content": response.assistant_next_response})

    display_structured_response(response)
    # Unfortunately not firing when pressing buttons...
    #render_feedback_buttons(len(st.session_state["generate_trace_ids"]) - 1)
