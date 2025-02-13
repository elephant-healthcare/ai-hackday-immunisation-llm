from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
import weave

from query_llm import query_llm
from llama_index.core import set_global_handler

from query_rag_llm import query_rag_llm

# Load environment variables from .env file
load_dotenv()

project = "ai_assistant_nurse_audio_app"
weave.init(project)
set_global_handler("wandb", run_args={"project": project})

# Set OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

st.title("I'm your Elephant AI assistant nurse Ella 👋 🐘")

recorded_audio = st.audio_input("Record the patient's issue")
uploaded_audio = st.file_uploader("...or upload an audio file of the patient's issue", type=["mp3"])

audio_data = recorded_audio if recorded_audio else uploaded_audio

if audio_data:
    st.write("**Audio input received:**")
    st.audio(audio_data)
    
    # Assuming audio_data is a BytesIO object
    audio_bytes = audio_data.getvalue()
    
    # Send audio to OpenAI Whisper
    transcription = OpenAI().audio.transcriptions.create(
        model="whisper-1",
        file=audio_data
    )
    
    # Display the transcribed text
    st.write("**Patient's issue transcript:**")
    st.write(transcription.text)
    
    # Define a priming prompt
    priming_prompt = "The following transcript will be a recording of a patient explaining their issue. I want you to return a response of clinical advice for the patient, but you will be providing this information to the clinician to relay to the patient. The response should be in the form of a list of questions to ask the patient to help us determine the cause of their issue."
    
    # Query the basic LLM with the transcribed text and priming prompt
    # response = query_llm(transcription.text, priming_prompt)
    
    # Query the RAG LLM with the transcribed text
    response = query_rag_llm(transcription.text)
    
    # Extract the text from the response object
    text = response.text  # Assuming 'response' has a 'text' attribute
    
    # Display the LLM response
    st.write("**Ella's response:**")
    st.markdown(text)


