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


def _bar(prob: float, width: int = 30) -> str:
    """Render a horizontal bar with █ blocks for a probability in [0,1]."""
    n = max(0, min(width, int(round(prob * width))))
    return "█" * n + "░" * (width - n)


def predict(audio_input):
    if audio_input is None:
        return "Upload audio or record from your mic, then click **Predict**."
    sr, audio = audio_input
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    tmp = "/tmp/_serdemo.wav"
    sf.write(tmp, audio, sr)
    results = get_pipeline()(tmp, top_k=5)

    # Format results as a Markdown table (avoids gr.Label schema introspection bug)
    lines = [
        "### 🎯 Predicted emotion",
        "",
        "| Emotion | Confidence | |",
        "|---|---|---|",
    ]
    for r in results:
        pct = f"{r['score'] * 100:5.1f}%"
        lines.append(f"| **{r['label']}** | {pct} | `{_bar(r['score'])}` |")
    lines.extend(
        [
            "",
            f"_Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})._ "
            "For custom-trained RoBERTa + wav2vec2 cross-attention models, "
            "see the [GitHub repo](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning).",
        ]
    )
    return "\n".join(lines)


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
            result_md = gr.Markdown("_Predictions will appear here._")

    btn.click(predict, inputs=audio, outputs=result_md)

    gr.Markdown(
        """
        ---
        Built by [Shahnawaz Khan](https://www.skakarh.com) · [GitHub](https://github.com/ShahnawazKakarh) · [LinkedIn](https://www.linkedin.com/in/skakarh) · [Products](https://www.skakarh.com/products)
        """
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
