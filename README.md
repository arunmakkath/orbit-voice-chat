# Orbit Voice Chat (Self-contained)

This app runs a voice-based assistant using a locally bundled version of Sesame CSM (based on Wav2Vec2).

## Features

- Upload voice (`.wav`)
- Generate audio embedding
- GPT-4-based contextual response

## Setup

1. Add your OpenAI API key to a `.env` file.
2. Run `streamlit run app/voice_chat_app.py`.

## Deployment

- Upload to GitHub
- Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud)