import streamlit as st
import tempfile
import os
import openai
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoProcessor, CSMModel
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🎤 Orbit Voice Chat", layout="centered")
st.title("🎤 Orbit Voice Chat Assistant (Sesame CSM)")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("sesame/CSM")
    model = CSMModel.from_pretrained("sesame/CSM")
    return processor, model

processor, model = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def record_audio(filename, duration=5, samplerate=16000):
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, recording, samplerate)

def get_embedding(audio_path):
    audio_input, sr = sf.read(audio_path)
    inputs = processor(audio=audio_input, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding

def generate_response(user_prompt, embedding):
    embedding_str = ", ".join([f"{x:.3f}" for x in embedding[0][:5]])
    prompt = f"User spoke, embedding summary: [{embedding_str}]"
    messages = [{"role": "system", "content": "You are a helpful assistant that replies based on voice context."}]
    for turn in st.session_state.chat_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content, prompt

duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

if st.button("Record & Talk"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        record_audio(tmpfile.name, duration)
        st.audio(tmpfile.name)
        embedding = get_embedding(tmpfile.name)
        reply, prompt = generate_response("User input from audio", embedding)
        st.success(reply)
        st.session_state.chat_history.append({"user": prompt, "bot": reply})

st.markdown("### Chat History")
for turn in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You**: {turn['user']}")
    st.markdown(f"**🤖 Bot**: {turn['bot']}")