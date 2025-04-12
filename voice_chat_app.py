import streamlit as st
import tempfile
import os
import openai
import torch
import soundfile as sf
from dotenv import load_dotenv
from csm import AutoProcessor, CSMModel

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🎤 Orbit Voice Chat", layout="centered")
st.title("🎤 Orbit Voice Chat Assistant (Local Sesame CSM)")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = CSMModel.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_embedding(audio_path):
    audio_input, sr = sf.read(audio_path)
    inputs = processor(audio=audio_input, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**inputs).mean(dim=1)
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

uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        audio_path = tmpfile.name
        st.audio(audio_path)
        embedding = get_embedding(audio_path)
        reply, prompt = generate_response("User input from audio", embedding)
        st.success(reply)
        st.session_state.chat_history.append({"user": prompt, "bot": reply})

st.markdown("### Chat History")
for turn in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You**: {turn['user']}")
    st.markdown(f"**🤖 Bot**: {turn['bot']}")