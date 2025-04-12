# Orbit Voice Chat (Streamlit + Sesame CSM)

This app allows you to upload a .wav voice file, processes it with Sesame CSM, and generates a GPT-4-based reply using audio context.

## Usage

1. Upload a `.wav` file in the browser.
2. The app computes embeddings via Sesame CSM.
3. A GPT-4 reply is generated based on the embedding.

## Deployment

Use [Streamlit Community Cloud](https://streamlit.io/cloud):

- Set `app/voice_chat_app.py` as the entry point.
- Add your OpenAI key in app secrets: `OPENAI_API_KEY`.

## Requirements

See `requirements.txt`. The CSM model is pulled directly from GitHub.