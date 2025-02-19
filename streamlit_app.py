from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
import weave

from query_rag_llm import create_query_rag_llm_v2

# Load environment variables from .env file
load_dotenv()

project = "ai-hackday-immunisation-nurse"
weave.init(project)

# Set OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def create_query_rag_llm():
    return create_query_rag_llm_v2()

st.session_state["create_query_rag_llm"] = create_query_rag_llm()

st.title("I'm your Elephant AI assistant nurse Ella 👋 🐘")

recorded_audio = st.audio_input("Record the patient's issue")
uploaded_audio = st.file_uploader("...or upload an audio file of the patient's issue", type=["mp3"])

audio_data = recorded_audio if recorded_audio else uploaded_audio

transcribe_prompt = "The audio is a recording of a patient explaining their issue. Please use awareness of what makes sense in a clinical setting to transcribe the audio."

@weave.op()
def transcribe_audio(audio_data):
    # Send audio to OpenAI Whisper
    transcription = OpenAI().audio.transcriptions.create(
        model="whisper-1",
        file=audio_data,
        prompt=transcribe_prompt
    )
    return transcription.text

if audio_data:
    st.write("**Audio input received:**")
    st.audio(audio_data)
    
    transcription = transcribe_audio(audio_data)
    # Display the transcribed text
    st.write("**Patient's issue transcript:**")
    st.write(transcription)
    
    # Define a priming prompt
    priming_prompt = "The following transcript will be a recording of a patient explaining their issue. I want you to return a response of clinical advice for the patient, but you will be providing this information to the clinician to relay to the patient. The response should be in the form of a list of questions to ask the patient to help us determine the cause of their issue."
        
    # Query the RAG LLM with the transcribed text
    response = response = st.session_state["create_query_rag_llm"](transcription)
    
    # Display the LLM response
    st.write("**Ella's response:**")
    st.markdown(str(response))

    # Sharing knowledge use by LLM
    container = st.container()
    container.status = st.status("**Ella's contextual knowledge:**")
    for idx, node in enumerate(response.source_nodes):
        container.status.write(f"**Document {idx} from {node.metadata['file_name']}**")
        container.status.markdown(node.text)
    container.status.update(state="complete")


