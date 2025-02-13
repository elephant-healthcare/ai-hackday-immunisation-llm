from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

st.title("🐘 AUDIO LLM CHATBOT INCOMING!!! 🐘")

recorded_audio = st.audio_input("Record or upload the patient's issue")
uploaded_audio = st.file_uploader("Or upload an audio file", type=["mp3"])

audio_data = recorded_audio if recorded_audio else uploaded_audio

if audio_data:
    st.write("Audio input received:")
    st.audio(audio_data)
    
    # Assuming audio_data is a BytesIO object
    audio_bytes = audio_data.getvalue()
    
    # Send audio to OpenAI Whisper
    transcription = OpenAI().audio.transcriptions.create(
        model="whisper-1",
        file=audio_data
    )
    
    # Display the transcribed text
    st.write("Transcribed Text:")
    st.write(transcription.text)


