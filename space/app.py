"""Gradio app for HuggingFace Spaces deployment.

Runs a pre-trained SER pipeline (no custom checkpoint needed) so the Space
boots in under a minute. To switch to a custom-trained checkpoint, see the
'Upgrading to a custom checkpoint' section in space/README_DEPLOY.md.
"""
from __future__ import annotations

import gradio as gr
import numpy as np
import soundfile as sf
from transformers import pipeline

# Pre-trained SER model from the SUPERB benchmark
MODEL_ID = "superb/wav2vec2-base-superb-er"

# Lazy-init so the Space boots fast and the model loads on the first request
_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("audio-classification", model=MODEL_ID)
    return _pipe


def predict(audio_input):
    if audio_input is None:
        return {}, "Upload audio or record from your mic, then click Predict."
    sr, audio = audio_input
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    tmp = "/tmp/_serdemo.wav"
    sf.write(tmp, audio, sr)
    results = get_pipeline()(tmp, top_k=5)
    probs = {r["label"]: float(r["score"]) for r in results}
    note = (
        f"Inference on `{MODEL_ID}`. "
        "For custom-trained RoBERTa + wav2vec2 cross-attention models, see the GitHub repo."
    )
    return probs, note


with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎙️ Speech Emotion Recognition

        Detect emotion in speech using transfer learning. Upload a `.wav` / `.mp3` / `.flac`
        or record from your microphone, then click **Predict**.

        - **Code & full results**: [ShahnawazKakarh/speech-emotion-recognition-transfer-learning](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning)
        - **Best published-style result** (multimodal cross-attention, speaker-independent split on RAVDESS): WF1 **0.728**
        - **This Space** runs a public pretrained baseline (`superb/wav2vec2-base-superb-er`) for instant inference.
        """
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(
                label="Audio input",
                sources=["upload", "microphone"],
                type="numpy",
            )
            btn = gr.Button("🔍 Predict", variant="primary")
        with gr.Column():
            label_out = gr.Label(label="Predicted emotion", num_top_classes=5)
            note_out = gr.Markdown()

    btn.click(predict, inputs=audio, outputs=[label_out, note_out])

    gr.Markdown(
        """
        ---
        Built by [Shahnawaz Khan](https://www.skakarh.com) · [GitHub](https://github.com/ShahnawazKakarh) · [LinkedIn](https://www.linkedin.com/in/skakarh) · [Products](https://www.skakarh.com/products)
        """
    )


if __name__ == "__main__":
    # show_api=False disables the /api/info endpoint, which has a known
    # 'bool' schema introspection bug in Gradio 5.x that floods logs.
    app.launch(show_api=False)
