# Quiz Shorts Generator

Generate vertical 9:16 quiz videos with AI voice-over optimized for YouTube Shorts.

## Features
- 1080×1920 and 720×1280 resolutions with safe zones for YouTube UI
- Background image support with blur/darken for legibility
- Typing animation for questions and sequential fade for options
- Answer reveal with green highlight and explanation slide
- Engagement and outro slides
- OpenAI or manual question entry
- TTS via OpenAI, gTTS or pyttsx3
- Adjustable FPS, typing speed, fade duration and segment holds
- Downloadable MP4 output

## Safe Zones
The app reserves the bottom 25% of the frame for YouTube interface elements. Main content is confined to roughly the top 8%–70%. Toggle the **Show safe-zone guides** option to visualize these regions.

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Secrets
Configure your OpenAI key in any of the following locations:
```toml
# .streamlit/secrets.toml examples
OPENAI_API_KEY = "sk-..."
openai_api_key = "sk-..."
[openai]
api_key = "sk-..."
```

## Deploy on Streamlit Cloud
1. Fork this repo to your GitHub account.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Add your OpenAI secret in the dashboard (key `OPENAI_API_KEY` or `openai_api_key`).

## TTS Modes
- **openai** – best quality, requires API key.
- **gtts** – free fallback using Google TTS.
- **pyttsx3** – local engine; may not work on Streamlit Cloud.

## Troubleshooting
- **ffmpeg errors**: ensure `imageio-ffmpeg` is installed and system FFmpeg is available.
- **TTS failures**: check API keys and network access. pyttsx3 may require audio drivers locally.
