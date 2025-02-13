from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
import weave

from query_llm import query_llm

from query_rag_llm import query_rag_llm

# Load environment variables from .env file
load_dotenv()

project = "ai_assistant_nurse_audio_app"
weave.init(project)

# Set OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

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
    
    # Query the basic LLM with the transcribed text and priming prompt
    # response = query_llm(transcription.text, priming_prompt).text
    
    # Query the RAG LLM with the transcribed text
    response = query_rag_llm(transcription)
    
    # Display the LLM response
    st.write("**Ella's response:**")
    st.markdown(str(response))


