import streamlit as st
import pandas as pd

st.title("🐘 AUDIO LLM CHATBOT INCOMING!!! 🐘")

recorded_audio = st.audio_input("Record or upload the patient's issue")
uploaded_audio = st.file_uploader("Or upload an audio file", type=["mp3"])

audio_data = recorded_audio if recorded_audio else uploaded_audio

if audio_data:
    st.write("Audio input received:")
    st.audio(audio_data)


