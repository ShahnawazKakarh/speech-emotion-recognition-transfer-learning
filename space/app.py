"""Gradio app for HuggingFace Spaces deployment.

Runs a pre-trained SER pipeline (no custom checkpoint needed) so the Space
boots in under a minute. To switch to a custom-trained checkpoint, see the
'Upgrading to a custom checkpoint' section in space/README_DEPLOY.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Workaround for known gradio_client bug:
#   TypeError: argument of type 'bool' is not iterable
#   in gradio_client.utils.get_type when schema is True/False
# This monkeypatch MUST run BEFORE `import gradio`.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils  # noqa: E402

_original_get_type = _gc_utils.get_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _original_get_type(schema)


_gc_utils.get_type = _safe_get_type

_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
# ---------------------------------------------------------------------------

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import pipeline  # noqa: E402

MODEL_ID = "superb/wav2vec2-base-superb-er"

# Color palette for emotion bars (consistent across re-renders)
EMOTION_COLORS = {
    "hap": "#22c55e",   # green — happy
    "neu": "#94a3b8",   # slate — neutral
    "ang": "#ef4444",   # red — angry
    "sad": "#3b82f6",   # blue — sad
}
DEFAULT_COLOR = "#a78bfa"  # purple fallback

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("audio-classification", model=MODEL_ID)
    return _pipe


def _render_results_html(results) -> str:
    """Render results as responsive HTML/CSS bars that scale to any viewport."""
    rows = []
    for r in results:
        label = r["label"]
        pct = r["score"] * 100
        color = EMOTION_COLORS.get(label, DEFAULT_COLOR)
        rows.append(
            f"""
            <div class="ser-row">
              <div class="ser-label">{label}</div>
              <div class="ser-track">
                <div class="ser-fill" style="width: {pct:.1f}%; background: {color};"></div>
              </div>
              <div class="ser-pct">{pct:.1f}%</div>
            </div>
            """
        )

    return f"""
    <style>
      .ser-results {{
        font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
        padding: 0.25rem 0;
      }}
      .ser-results h3 {{
        margin: 0 0 0.75rem 0;
        font-size: 1.05rem;
        font-weight: 600;
      }}
      .ser-row {{
        display: grid;
        grid-template-columns: 56px 1fr 60px;
        gap: 12px;
        align-items: center;
        margin-bottom: 8px;
      }}
      .ser-label {{
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
      }}
      .ser-track {{
        height: 18px;
        border-radius: 6px;
        background: rgba(148, 163, 184, 0.18);
        overflow: hidden;
      }}
      .ser-fill {{
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease-out;
      }}
      .ser-pct {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-size: 0.9rem;
        opacity: 0.85;
      }}
      .ser-note {{
        margin-top: 1rem;
        font-size: 0.85rem;
        opacity: 0.75;
        line-height: 1.5;
      }}
      @media (max-width: 480px) {{
        .ser-row {{ grid-template-columns: 44px 1fr 52px; gap: 8px; }}
        .ser-label {{ font-size: 0.85rem; }}
        .ser-pct {{ font-size: 0.82rem; }}
      }}
    </style>
    <div class="ser-results">
      <h3>🎯 Predicted emotion</h3>
      {''.join(rows)}
      <div class="ser-note">
        Model: <a href="https://huggingface.co/{MODEL_ID}"><code>{MODEL_ID}</code></a>.
        For custom-trained RoBERTa + wav2vec2 cross-attention models, see the
        <a href="https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning">GitHub repo</a>.
      </div>
    </div>
    """


def predict(audio_input):
    if audio_input is None:
        return "<p style='opacity:0.7'>Upload audio or record from your mic, then click <b>Predict</b>.</p>"
    sr, audio = audio_input
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    tmp = "/tmp/_serdemo.wav"
    sf.write(tmp, audio, sr)
    results = get_pipeline()(tmp, top_k=5)
    return _render_results_html(results)


# Custom CSS: stack the input/output row vertically on narrow viewports
APP_CSS = """
.gradio-container { max-width: 920px !important; }

@media (max-width: 640px) {
  .gradio-container { padding: 8px !important; }
  /* Stack the main row vertically on mobile */
  #ser-main-row { flex-direction: column !important; }
  #ser-main-row > .gr-column { min-width: 0 !important; }
}

/* Tighten markdown spacing in the intro */
.ser-intro p { margin: 0.4rem 0; }
.ser-intro ul { margin: 0.4rem 0; padding-left: 1.2rem; }
"""


with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft(), css=APP_CSS) as app:
    gr.Markdown(
        """
        # 🎙️ Speech Emotion Recognition

        Detect emotion in speech using transfer learning.
        Upload an audio clip (`.wav`, `.mp3`, `.flac`) or record from your mic — then click **Predict**.

        - **Code & full results**: [ShahnawazKakarh/speech-emotion-recognition-transfer-learning](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning)
        - **Best published-style result** (multimodal cross-attention, speaker-independent split on RAVDESS): WF1 **0.728**
        - **This Space** runs a public pretrained baseline (`superb/wav2vec2-base-superb-er`) for instant inference.
        """,
        elem_classes=["ser-intro"],
    )
    with gr.Row(elem_id="ser-main-row"):
        with gr.Column(scale=1, min_width=280):
            audio = gr.Audio(
                label="Audio input",
                sources=["upload", "microphone"],
                type="numpy",
            )
            btn = gr.Button("🔍 Predict", variant="primary", size="lg")
        with gr.Column(scale=1, min_width=280):
            result_html = gr.HTML(
                value="<p style='opacity:0.7'>Predictions will appear here.</p>",
                label="Predicted emotion",
            )

    btn.click(predict, inputs=audio, outputs=result_html)

    gr.Markdown(
        """
        ---
        Built by [Shahnawaz Khan](https://www.skakarh.com) · [GitHub](https://github.com/ShahnawazKakarh) · [LinkedIn](https://www.linkedin.com/in/skakarh) · [Products](https://www.skakarh.com/products)
        """
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
