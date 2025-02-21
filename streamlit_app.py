from pathlib import Path
import uuid
from dotenv import load_dotenv
import os

import streamlit as st
from openai import OpenAI
import weave

from query_rag_llm import Response, RagModel

TRANSCRIPTION_MODEL = "whisper-1"
TRANSCRIPTION_PROMPT = "The audio is a recording of a patient explaining their issue. Please use awareness of what makes sense in a clinical setting to transcribe the audio."
CHAT_MODEL = "mistral-large-latest"

# Load environment variables from .env file
load_dotenv()

project = "ai-hackday-immunisation-nurse"
weave.init(project)

# Set OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def init_states():
    """Set up session_state keys if they don't exist yet."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "calls" not in st.session_state:
        st.session_state["calls"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "rag_model" not in st.session_state:
        st.session_state["rag_model"] = RagModel(name=CHAT_MODEL, chat_llm=CHAT_MODEL)


def render_feedback_buttons(call_idx):
    """Renders thumbs up/down and text feedback for the call.
    Adapted from https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
    """

    col1, col2, col3 = st.columns([1, 1, 4])

    print("render_feedback_buttons", len(st.session_state.calls), call_idx)

    with col1:
        if st.button("👍", key=f"thumbs_up_{call_idx}"):
            st.session_state.calls[call_idx].feedback.add_reaction("👍")

    with col2:
        if st.button("👎", key=f"thumbs_down_{call_idx}"):
            st.session_state.calls[call_idx].feedback.add_reaction("👎")


@weave.op()
def transcribe_audio(audio_data):
    # Send audio to OpenAI Whisper
    transcription = OpenAI().audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=audio_data,
        prompt=TRANSCRIPTION_PROMPT
    )
    return transcription.text

# https://github.com/remsky/Kokoro-FastAPI/blob/b00c9ec28df0fd551ae25108a986e04d29a54f2e/api/src/core/openai_mappings.json#L4
KOKORO_VOICE = "af_v0nicole" # af_sarah
KOKORO_VOICE = "bf_v0emma+ff_siwis"

def text_to_audio(text) -> bytes:
    # Free for now!
    client = OpenAI(base_url="https://api.kokorotts.com/v1", api_key="not-needed")
    response = client.audio.speech.create(
        model="kokoro",  # Not used but required for compatibility
        voice=KOKORO_VOICE,
        input=text,
        response_format="mp3"
    )
    return response.read()

def display_and_transcribe_audio(audio_data) -> str:
    st.write("**Audio input received:**", )
    st.audio(audio_data)
    
    transcription = transcribe_audio(audio_data)
    # Display the transcribed text
    with st.chat_message("human", avatar=patient):
        st.write(transcription)

    return transcription

def display_contextual_knowledge(response: Response):
    container = st.container()
    container.status = st.status("**Ella's contextual knowledge**")
    for idx, node in enumerate(response.source_nodes):
        container.status.write(f"**Document {idx} from {node.metadata['file_name']}**")
        container.status.markdown(node.text)
    container.status.update(state="complete")

def display_assistant_message(display_user_message=False):
    """Displays the conversation stored in st.session_state.messages with feedback buttons"""

    if len(st.session_state.messages) > 0:
        user_message, assistant_message = st.session_state.messages[-2:]

        if display_user_message:
            with st.chat_message("human", avatar=patient):
                st.markdown(user_message["content"])

        with st.chat_message("ai", avatar=logo):
            st.write("**Ella's response**")
            st.markdown(assistant_message["content"])

        render_feedback_buttons(len(st.session_state.calls) - 1)


def query_assistant(user_input: str) -> Response:

    # Attach Weave attributes for tracking of conversation instances
    with weave.attributes(
        {"session": st.session_state["session_id"], "env": "prod"}
    ):
        rag_model = st.session_state["rag_model"]

        # Calling the LLM through the weave decorated function to retrieve the weave call object
        # self has to be passed as first argument as we are calling a decorated function
        # https://weave-docs.wandb.ai/reference/gen_notebooks/feedback_prod/
        response, call = rag_model.predict.call(rag_model, query=user_input)

        # Store the weave call object to link feedback to the specific response
        st.session_state.calls.append(call)

        # Store the assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )
        return response


if __name__ == "__main__":

    logo = Path("elephant.png")
    patient = Path("patient_accurx.png")
    init_states()

    st.title("I'm your AI nurse Ella 👋 🐘")

    audio_tab, text_tab = st.tabs(
        ["Audio", "Text"],
        )

    with audio_tab:

        recorded_audio = st.audio_input("Record the patient's issue")
        uploaded_audio = st.file_uploader("...or upload an audio file of the patient's issue", type=["mp3"])
        audio_data = recorded_audio if recorded_audio else uploaded_audio

        if audio_data:
            transcription = display_and_transcribe_audio(audio_data)
        
            # Query the RAG LLM with the transcribed text
            response = st.session_state["rag_model"].predict(transcription)
            
            # Display the LLM context and then the response
            with st.chat_message("ai", avatar=logo):
                display_contextual_knowledge(response)

            with st.chat_message("ai", avatar=logo):
                st.markdown(str(response))

            # Play audio last as ~5-10s delay
            st.audio(text_to_audio(str(response)), autoplay=True)

    with text_tab:
        if user_input := st.chat_input("Ask a question:"):
            # Immediately render new user message
            with st.chat_message("human", avatar=patient):
                st.markdown(user_input)
            # And also save message in session (for so that it shows in the chat history on rerenders)
            st.session_state.messages.append({"role": "user", "content": user_input})

            response = query_assistant(user_input)

            with st.chat_message("ai", avatar=logo):
                display_contextual_knowledge(response)
     
        # empty to start with, then show assistant message, then show both messages on rerender
        display_assistant_message(display_user_message=user_input is None)
