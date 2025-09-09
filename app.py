import os
import io
import json
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from moviepy.editor import (AudioFileClip, ImageSequenceClip,
                            concatenate_audioclips)
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_SDK = True
except Exception:
    OPENAI_SDK = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_openai_api_key() -> Optional[str]:
    """Fetch OpenAI key from env or Streamlit secrets."""
    candidates = [
        os.environ.get("OPENAI_API_KEY"),
        st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None,
        st.secrets.get("openai_api_key") if "openai_api_key" in st.secrets else None,
    ]
    if "openai" in st.secrets and isinstance(st.secrets["openai"], dict):
        candidates.append(st.secrets["openai"].get("api_key"))
    for key in candidates:
        if key:
            return key
    return None


def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if font.getlength(test) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def draw_multiline(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont,
                   box: tuple, fill: str = "white", align: str = "center"):
    lines = wrap_text(text, font, box[2] - box[0])
    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + 5
    total_height = len(lines) * line_height
    y = box[1] + (box[3] - box[1] - total_height) // 2
    for line in lines:
        w = font.getlength(line)
        if align == "center":
            x = box[0] + (box[2] - box[0] - w) // 2
        else:
            x = box[0]
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


@dataclass
class Layout:
    width: int
    height: int
    content_top: int
    content_bottom: int
    bottom_safe: int

    @classmethod
    def create(cls, width: int, height: int) -> "Layout":
        return cls(width, height, int(height * 0.08), int(height * 0.70),
                   int(height * 0.75))


def get_background(size: tuple, uploaded) -> Image.Image:
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB").resize(size)
        img = img.filter(ImageFilter.GaussianBlur(6))
        img = ImageEnhance.Brightness(img).enhance(0.4)
    else:
        # dark vertical gradient
        img = Image.new("RGB", size, "black")
        top = Image.new("RGB", size, "#222")
        mask = Image.linear_gradient("L").resize(size)
        img = Image.composite(top, img, mask)
    return img


def apply_guides(img: Image.Image, layout: Layout) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, layout.content_top, layout.width, layout.content_bottom],
                   outline=(255, 255, 0, 255), width=4)
    draw.rectangle([0, layout.bottom_safe, layout.width, layout.height],
                   outline=(255, 0, 0, 255), width=4)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# -----------------------------------------------------------------------------
# Rendering functions
# -----------------------------------------------------------------------------


def render_title_frame(bg: Image.Image, layout: Layout, title: str, day: str,
                       quiz_no: int, show_guides: bool) -> Image.Image:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    title_font = load_font(int(layout.width * 0.08))
    sub_font = load_font(int(layout.width * 0.05))
    box = (0, layout.content_top, layout.width, layout.content_bottom)
    draw_multiline(draw, title, title_font, box)
    subtitle = f"{day} • Quiz #{quiz_no}"
    sub_lines = wrap_text(subtitle, sub_font, box[2] - box[0])
    y = layout.content_bottom - (len(sub_lines) * (sub_font.getbbox("Ay")[3] - sub_font.getbbox("Ay")[1] + 5))
    for line in sub_lines:
        w = sub_font.getlength(line)
        draw.text(((layout.width - w) / 2, y), line, font=sub_font, fill="white")
        y += sub_font.getbbox("Ay")[3] - sub_font.getbbox("Ay")[1] + 5
    if show_guides:
        img = apply_guides(img, layout)
    return img


def render_question_frame(bg: Image.Image, layout: Layout, text: str,
                          show_guides: bool) -> Image.Image:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    q_font = load_font(int(layout.width * 0.06))
    box = (int(layout.width * 0.05), layout.content_top,
           int(layout.width * 0.95), layout.content_bottom)
    draw_multiline(draw, text, q_font, box, align="left")
    if show_guides:
        img = apply_guides(img, layout)
    return img


def render_options_frame(bg: Image.Image, layout: Layout, question: str,
                         options: List[str], alphas: List[int],
                         show_guides: bool) -> Image.Image:
    img = render_question_frame(bg, layout, question, False)
    draw = ImageDraw.Draw(img)
    o_font = load_font(int(layout.width * 0.05))
    box = (int(layout.width * 0.1), int(layout.height * 0.45),
           int(layout.width * 0.9), layout.content_bottom)
    line_height = o_font.getbbox("Ay")[3] - o_font.getbbox("Ay")[1] + 10
    y = box[1]
    letters = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        line = f"{letters[i]}. {opt}"
        w = o_font.getlength(line)
        x = box[0]
        draw.text((x, y), line, font=o_font,
                  fill=(255, 255, 255, alphas[i]))
        y += line_height
    if show_guides:
        img = apply_guides(img, layout)
    return img


def render_answer_frame(bg: Image.Image, layout: Layout, question: str,
                        options: List[str], correct: int, explanation: str,
                        show_guides: bool) -> Image.Image:
    img = render_options_frame(bg, layout, question, options,
                               [255] * 4, False)
    draw = ImageDraw.Draw(img)
    o_font = load_font(int(layout.width * 0.05))
    line_height = o_font.getbbox("Ay")[3] - o_font.getbbox("Ay")[1] + 10
    y = int(layout.height * 0.45) + correct * line_height
    draw.rectangle([int(layout.width * 0.1) - 10, y - 5,
                    int(layout.width * 0.9) + 10, y + line_height - 5],
                   fill=(0, 100, 0, 160))
    img = render_options_frame(img, layout, question, options,
                               [255] * 4, False)
    e_font = load_font(int(layout.width * 0.045))
    box = (int(layout.width * 0.05),
           int(layout.height * 0.55),
           int(layout.width * 0.95), layout.content_bottom)
    draw_multiline(draw, explanation, e_font, box, align="left")
    if show_guides:
        img = apply_guides(img, layout)
    return img


def render_engagement_frame(bg: Image.Image, layout: Layout,
                             show_guides: bool) -> Image.Image:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(int(layout.width * 0.08))
    box = (0, layout.content_top, layout.width, layout.content_bottom)
    draw_multiline(draw, "COMMENT YOUR ANSWER BELOW!", font, box)
    if show_guides:
        img = apply_guides(img, layout)
    return img


def render_outro_frame(bg: Image.Image, layout: Layout,
                       show_guides: bool) -> Image.Image:
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    font = load_font(int(layout.width * 0.08))
    box = (0, layout.content_top, layout.width, layout.content_bottom)
    draw_multiline(draw, "LIKE • SHARE • SUBSCRIBE", font, box)
    if show_guides:
        img = apply_guides(img, layout)
    return img


# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------


def tts_save_openai(text: str, path: str, api_key: str):
    client = OpenAI(api_key=api_key)
    try:
        with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice="alloy", input=text) as r:
            r.stream_to_file(path)
    except Exception:
        with client.audio.speech.with_streaming_response.create(
                model="tts-1", voice="alloy", input=text) as r:
            r.stream_to_file(path)


def tts_save_gtts(text: str, path: str):
    tts = gTTS(text)
    tts.save(path)


def tts_save_pyttsx3(text: str, path: str):
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()


def tts_save(text: str, path: str, mode: str, api_key: Optional[str]):
    if mode == "openai" and api_key:
        tts_save_openai(text, path, api_key)
    elif mode == "openai" and not api_key:
        st.warning("No OpenAI key found, falling back to gTTS")
        tts_save_gtts(text, path)
    elif mode == "gtts":
        tts_save_gtts(text, path)
    else:
        tts_save_pyttsx3(text, path)


# -----------------------------------------------------------------------------
# OpenAI Question Generation
# -----------------------------------------------------------------------------


def generate_question(topic: str, difficulty: str, api_key: str):
    client = OpenAI(api_key=api_key)
    prompt = (
        "Create a multiple choice question on the topic '" + topic +
        "' with difficulty '" + difficulty +
        "'. Respond in JSON with keys question, options (4 items), correct_index, explanation."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    content = res.choices[0].message.content
    return json.loads(content)


# -----------------------------------------------------------------------------
# Video assembly
# -----------------------------------------------------------------------------


def assemble_video(frames: List[Image.Image], audio_paths: List[str], fps: int,
                   out_path: str):
    clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
    audioclips = [AudioFileClip(p) for p in audio_paths]
    audio = concatenate_audioclips(audioclips)
    clip = clip.set_audio(audio)
    clip.write_videofile(out_path, fps=fps, codec="libx264",
                         audio_codec="aac", verbose=False, logger=None)
    for c in audioclips:
        c.close()
    return out_path


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------


def main():
    st.title("Quiz Shorts Generator")
    api_key = _get_openai_api_key()

    resolution = st.selectbox("Resolution", ["1080x1920", "720x1280"])
    width, height = (1080, 1920) if resolution == "1080x1920" else (720, 1280)
    layout = Layout.create(width, height)

    show_guides = st.checkbox("Show safe-zone guides", value=False)
    bg_upload = st.file_uploader("Background image", type=["png", "jpg", "jpeg"])

    quiz_title = st.text_input("Quiz Title", "Daily Quiz")
    day_label = st.text_input("Day label", "Day 1")
    quiz_number = st.number_input("Quiz #", min_value=1, value=1)

    source = st.radio("Question source", ["Manual", "OpenAI"], horizontal=True)
    if source == "OpenAI":
        topic = st.text_input("Topic")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
        if st.button("Generate question"):
            if not api_key:
                st.error("OpenAI API key required")
            else:
                try:
                    data = generate_question(topic, difficulty, api_key)
                    st.session_state.qdata = data
                except Exception as e:
                    st.error("Failed to generate question")
                    st.exception(e)
        data = st.session_state.get("qdata", {
            "question": "",
            "options": ["", "", "", ""],
            "correct_index": 0,
            "explanation": "",
        })
        question = st.text_area("Question", data["question"])
        options = []
        for i in range(4):
            options.append(st.text_input(f"Option {chr(65+i)}", data["options"][i]))
        correct_index = st.number_input("Correct index", 0, 3, data["correct_index"])
        explanation = st.text_area("Explanation", data["explanation"])
    else:
        question = st.text_area("Question")
        options = [st.text_input(f"Option {chr(65+i)}") for i in range(4)]
        correct_index = st.number_input("Correct index", 0, 3, 0)
        explanation = st.text_area("Explanation")

    tts_mode = st.selectbox("TTS Mode", ["openai", "gtts", "pyttsx3"])
    if tts_mode == "openai" and not api_key:
        st.warning("No OpenAI key detected; switching to gTTS")
        tts_mode = "gtts"

    fps = st.slider("FPS", 20, 40, 30)
    typing_speed = st.slider("Typing speed (chars/frame)", 1, 10, 2)
    fade_frames = st.slider("Option fade frames", 1, 30, 10)
    hold_title = st.slider("Title hold (s)", 1, 5, 2)
    post_options_hold = st.slider("Post options hold (s)", 1, 5, 1)
    answer_hold = st.slider("Answer hold (s)", 1, 5, 2)
    engagement_hold = st.slider("Engagement hold (s)", 1, 3, 1)
    outro_hold = st.slider("Outro hold (s)", 1, 3, 1)

    bg = get_background((width, height), bg_upload)

    if st.button("Build Video"):
        try:
            frames: List[Image.Image] = []
            # Title frames
            for _ in range(int(hold_title * fps)):
                frames.append(
                    render_title_frame(bg, layout, quiz_title, day_label,
                                       quiz_number, show_guides))
            # Question typing frames
            for i in range(1, len(question) + 1, typing_speed):
                frames.append(
                    render_question_frame(bg, layout, question[:i], show_guides))
            # Options fade in
            alphas = [0, 0, 0, 0]
            for f in range(fade_frames * 4):
                idx = f // fade_frames
                prog = (f % fade_frames + 1) / fade_frames
                alphas[idx] = int(255 * prog)
                frames.append(render_options_frame(bg, layout, question,
                                                   options, alphas, show_guides))
            for _ in range(int(post_options_hold * fps)):
                frames.append(render_options_frame(bg, layout, question,
                                                   options, [255]*4, show_guides))
            # Answer
            for _ in range(int(answer_hold * fps)):
                frames.append(render_answer_frame(bg, layout, question,
                                                  options, correct_index,
                                                  explanation, show_guides))
            # Engagement
            for _ in range(int(engagement_hold * fps)):
                frames.append(render_engagement_frame(bg, layout, show_guides))
            # Outro
            for _ in range(int(outro_hold * fps)):
                frames.append(render_outro_frame(bg, layout, show_guides))

            with tempfile.TemporaryDirectory() as tmp:
                texts = [
                    f"{quiz_title}. {day_label}. Quiz number {quiz_number}.",
                    question,
                ] + [f"Option {chr(65+i)}. {opt}" for i, opt in enumerate(options)] + [
                    f"The correct answer is option {chr(65+correct_index)}. {explanation}",
                    "Comment your answer below!",
                    "Like, share, subscribe.",
                ]
                audio_paths = []
                for idx, txt in enumerate(texts):
                    ext = ".mp3" if tts_mode != "pyttsx3" else ".wav"
                    ap = os.path.join(tmp, f"seg{idx}{ext}")
                    tts_save(txt, ap, tts_mode, api_key)
                    audio_paths.append(ap)
                out_path = os.path.join(tmp, "quiz.mp4")
                assemble_video(frames, audio_paths, fps, out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Download MP4", f, "quiz.mp4",
                                        mime="video/mp4")
                st.video(out_path)
        except Exception as e:
            st.error("Failed to build video")
            st.exception(e)

    # Diagnostics
    with st.expander("Diagnostics", expanded=False):
        st.write(f"Build: {datetime.utcnow().isoformat()}")
        st.write(f"OpenAI SDK installed: {OPENAI_SDK}")
        key = _get_openai_api_key()
        masked = key[-4:] if key else "None"
        st.write(f"OpenAI key detected: {'yes' if key else 'no'} ({masked})")
        st.write(f"gTTS available: {GTTS_AVAILABLE}")
        st.write(f"pyttsx3 available: {PYTTSX3_AVAILABLE}")


if __name__ == "__main__":
    main()
