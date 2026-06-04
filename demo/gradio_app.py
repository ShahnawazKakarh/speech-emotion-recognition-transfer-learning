"""Gradio demo: upload a .wav file, see predicted emotion + per-class probabilities.

Usage:
    python demo/gradio_app.py --checkpoint outputs/multimodal_ravdess/last.ckpt \
                              --config configs/multimodal_ravdess.yaml

For a quick out-of-the-box demo without training, run with --pretrained:
    python demo/gradio_app.py --pretrained
which uses a pre-trained SER model from HuggingFace.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch


def build_local_predictor(checkpoint: str, config: str):
    """Predictor backed by a checkpoint trained in this repo."""
    from src.data.meld import MELD_EMOTIONS
    from src.data.ravdess import RAVDESS_EMOTIONS
    from src.models.lightning_module import SERLightningModule
    from src.train import load_config

    cfg = load_config(config)
    dataset = cfg["dataset"]
    if dataset == "ravdess":
        label_names = list(RAVDESS_EMOTIONS.values())
    elif dataset == "meld":
        label_names = MELD_EMOTIONS
    else:
        label_names = [f"class_{i}" for i in range(cfg["model"]["num_classes"])]

    # Load to CPU first (avoids PyTorch-MPS "Unaligned blit request" bug),
    # then move to the best available device for inference.
    model = SERLightningModule.load_from_checkpoint(
        checkpoint,
        cfg=cfg,
        label_names=label_names,
        map_location="cpu",
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print(f"[demo] Model device: {device}")

    sample_rate = cfg["data"].get("sample_rate", 16000)
    max_sec = cfg["data"].get("max_audio_seconds", 8.0)
    max_samples = int(sample_rate * max_sec)

    # Optional ASR for the text branch
    transcriber = None
    if cfg["model"]["type"] in ("text_only", "multimodal"):
        from src.asr.whisper_transcribe import get_default_transcriber

        transcriber = get_default_transcriber("base")

    def predict(audio_input):
        if audio_input is None:
            return {}, ""
        sr, audio = audio_input
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Pad/truncate
        if len(audio) > max_samples:
            start = (len(audio) - max_samples) // 2
            audio = audio[start : start + max_samples]
        else:
            pad = max_samples - len(audio)
            audio = np.pad(audio, (pad // 2, pad - pad // 2))

        text = transcriber.transcribe(audio) if transcriber else ""

        batch = {
            "audio": torch.from_numpy(audio).unsqueeze(0).float().to(device),
            "text": [text],
        }
        with torch.no_grad():
            logits = model.model(batch)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {name: float(p) for name, p in zip(label_names, probs)}, text

    return predict


def build_pretrained_predictor():
    """Fallback predictor using a public HF SER model — for instant demo."""
    from transformers import pipeline

    pipe = pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
    )

    def predict(audio_input):
        if audio_input is None:
            return {}, "(pretrained model: no transcript)"
        sr, audio = audio_input
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Pipeline handles resampling
        tmp = "/tmp/_gradio_sample.wav"
        sf.write(tmp, audio, sr)
        results = pipe(tmp, top_k=5)
        return {r["label"]: float(r["score"]) for r in results}, "(pretrained model: no transcript)"

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use a public HF SER model")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.pretrained or not (args.checkpoint and args.config):
        print("[demo] Using pretrained HF model (superb/wav2vec2-base-superb-er)")
        predict_fn = build_pretrained_predictor()
    else:
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        print(f"[demo] Loading checkpoint: {args.checkpoint}")
        predict_fn = build_local_predictor(args.checkpoint, args.config)

    with gr.Blocks(title="Speech Emotion Recognition — Transfer Learning") as app:
        gr.Markdown(
            "# 🎙️ Speech Emotion Recognition\n"
            "Upload a `.wav` or `.mp3`, or record from your mic, to predict the speaker's emotion.\n\n"
            "Built with transfer learning over wav2vec2 / WavLM + RoBERTa. "
            "See the [repo](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning) "
            "for training details."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="Audio input", sources=["upload", "microphone"], type="numpy")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Emotion probabilities", num_top_classes=5)
                text_out = gr.Textbox(label="ASR transcript (text branch input)")

        btn.click(predict_fn, inputs=audio, outputs=[label_out, text_out])

    app.launch(share=args.share)


if __name__ == "__main__":
    main()
