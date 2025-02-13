import streamlit as st
import pandas as pd
from openai import OpenAI
import openai

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
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_bytes,
        file_type="mp3"
    )
    
    # Display the transcribed text
    st.write("Transcribed Text:")
    st.write(response['text'])


