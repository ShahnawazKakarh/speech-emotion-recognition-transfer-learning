---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
short_description: Detect emotion in speech using transfer learning (wav2vec2)
tags:
  - speech-emotion-recognition
  - wav2vec2
  - transfer-learning
  - audio-classification
  - emotion-recognition
---

# 🎙️ Speech Emotion Recognition

Live demo for the [speech-emotion-recognition-transfer-learning](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning) project — comparing text-only, audio-only, and multimodal transfer learning approaches for SER.

Upload a `.wav` / `.mp3` / `.flac` file or record from your microphone to see the predicted emotion and per-class probabilities.

**Backed by**: a public wav2vec2 model fine-tuned on the SUPERB ER benchmark. The actual research code, configs, custom-trained checkpoints, and full results live in the [GitHub repo](https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning).

**Repo highlights (RAVDESS, speaker-independent split)**:
- Multimodal cross-attention (RoBERTa + wav2vec2): **WF1 0.728**
- Audio-only (wav2vec2-base): WF1 0.659
- Text-only (RoBERTa-base): WF1 0.031 (deliberate ablation)

Multimodal beats audio-only by +6.9 pp WF1 on unseen speakers, with a +57 pp jump on the *neutral* class.

---

Built by [Shahnawaz Khan](https://www.skakarh.com) · [GitHub](https://github.com/ShahnawazKakarh) · [LinkedIn](https://www.linkedin.com/in/skakarh)
