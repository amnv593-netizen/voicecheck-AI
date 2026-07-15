import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import pickle
import os

@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_models()

st.set_page_config(page_title="VoiceCheck AI", page_icon="🎙️")
st.title("🎙️ VoiceCheck AI")
st.subheader("AI-Powered Voice Fake News Detector")
st.write("Upload a voice message and our AI will tell you if it is Fake or Real news!")
st.divider()

audio_file = st.file_uploader("📂 Upload your voice clip", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())
    filename = audio_file.name
    if filename.endswith(".mp3") or filename.endswith(".m4a"):
        sound = AudioSegment.from_file("temp_audio")
        sound.export("temp_audio.wav", format="wav")
        audio_path = "temp_audio.wav"
    else:
        audio_path = "temp_audio"
    st.info("🔄 Converting voice to text...")
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data, language="en-IN")
        except:
            text = r.recognize_google(audio_data, language="hi-IN")
        st.success("✅ Voice converted to text!")
        st.write("**Transcript:**", text)
        st.divider()
        vec = tfidf.transform([text])
        result = model.predict(vec)[0]
        confidence = round(model.predict_proba(vec).max() * 100, 2)
        if result == 1:
            st.error(f"⚠️ FAKE NEWS DETECTED — Confidence: {confidence}%")
        else:
            st.success(f"✅ REAL NEWS — Confidence: {confidence}%")
    except Exception as e:
        st.error(f"Could not process audio: {str(e)}")

st.divider()
st.caption("VoiceCheck AI — AI and ML Project")
