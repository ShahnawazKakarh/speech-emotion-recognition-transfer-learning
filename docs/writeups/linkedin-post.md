# LinkedIn post — Speech Emotion Recognition project announcement

**Suggested headline image**: side-by-side bar chart of WF1 across audio / text / multimodal on RAVDESS and MELD, with the surprising MELD result highlighted. Same chart can double as the blog post's OG image.

---

## Version A — long-form (~280 words, the contrarian hook)

I just shipped a Speech Emotion Recognition project, and the headline finding is the opposite of what I expected:

**Multimodal isn't always better.**

I trained three models — text-only (RoBERTa), audio-only (wav2vec2 / WavLM), and multimodal cross-attention — on two standard SER datasets, with the same architecture and pipeline.

📊 **RAVDESS (clean lab speech, single speaker, 2 fixed sentences)**
→ Multimodal beats audio-only by **+6.9 pp weighted F1**
→ Biggest win: neutral class gets **+57 pp F1** on unseen speakers

📊 **MELD (Friends TV dialogue, multi-party, noisy)**
→ Text-only beats multimodal by **−1.9 pp weighted F1**
→ Audio branch collapses on rare classes (F1 = 0.00 for surprise, fear, disgust)

Why? **Modality complementarity.** RAVDESS audio is rich and text is degenerate (same 2 sentences in every emotion) → fusion extracts genuine signal. MELD text is rich (real dialogue content) and audio is noisy (laugh tracks, music, multi-speaker scenes) → fusion drags the strong text representation toward the noisier audio one.

The practical takeaway: evaluate unimodal baselines on your domain before paying the multimodal inference cost. On MELD, a 124M-parameter text-only model outperforms a 221M-parameter multimodal one. That's a real efficiency win the "multimodal always wins" framing tends to hide.

Also live:
🤗 HuggingFace Space demo (upload audio, see predictions): https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition
💻 GitHub repo (configs, results, reproducible runs): https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning
📝 Full writeup on the blog: https://www.skakarh.com/blog/multimodal-speech-emotion-recognition-ravdess-meld/

Next: cross-lingual SER for Urdu, Hindi, Punjabi using XLM-R + multilingual wav2vec2. If you're working on emotion AI, multimodal fusion, or low-resource speech, I'd love to compare notes.

#SpeechEmotionRecognition #MachineLearning #DeepLearning #MultimodalAI #PyTorch #HuggingFace #ResearchInProgress

---

## Version B — short / mobile-friendly (~150 words)

Just shipped a Speech Emotion Recognition project with a result I wasn't expecting:

📊 RAVDESS → **multimodal wins** (+6.9 pp WF1 over audio-only, +57 pp F1 on the neutral class)
📊 MELD → **text-only wins** (−1.9 pp WF1 under multimodal)

Same architecture, same pipeline, opposite verdicts.

The lesson isn't about architecture — it's about **modality complementarity**. Multimodal fusion helps when both modalities carry independent signal. It hurts when one is strong and the other is noise.

For practitioners: evaluate unimodal baselines before paying multimodal inference cost. On MELD, a 124M-parameter text model beats a 221M-parameter fusion model.

🤗 Live demo: https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition
💻 Code: https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning
📝 Blog: https://www.skakarh.com/blog/multimodal-speech-emotion-recognition-ravdess-meld/

Next: cross-lingual SER for Urdu / Hindi / Punjabi.

#SpeechEmotionRecognition #MultimodalAI #PyTorch #ML

---

## Version C — comment-bait variant (~200 words, ends on question)

"Multimodal models always beat unimodal ones."

I believed this too. Then I trained text-only, audio-only, and multimodal SER on two emotion datasets, same architecture, same pipeline.

📊 RAVDESS → multimodal wins by +6.9 pp
📊 MELD → multimodal LOSES to text-only by −1.9 pp

The pattern is clear in hindsight:

✅ RAVDESS audio = rich prosody, text = useless (2 fixed sentences) → fusion helps
❌ MELD audio = noisy TV-show clips, text = real dialogue → fusion hurts

A 221M-parameter multimodal model lost to a 124M-parameter text-only one because the audio branch (collapsing on 3 of 7 emotion classes at F1 = 0.00) dragged the strong text representation down.

The takeaway for anyone building emotion AI in production: **evaluate unimodal baselines on your actual domain first.** The "multimodal always wins" framing hides real efficiency wins and sometimes real accuracy losses.

Live demo + code + full results all open:
🤗 https://huggingface.co/spaces/Shahnawazkakarh/speech-emotion-recognition
💻 https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning

Curious what others have seen — when has multimodal failed for you? Or have you always seen it win?

#SpeechEmotionRecognition #MultimodalAI #PyTorch #HuggingFace #ML

---

## Posting strategy

**When to post**: Tuesday or Wednesday, 9-10am AEST (catches both Asia-Pacific and US engagement windows).

**Hashtag mix**: keep it 5-7 hashtags max. The included list balances reach (#MachineLearning, #DeepLearning) with niche-targeting (#SpeechEmotionRecognition, #MultimodalAI). Avoid #AI alone — too broad.

**Image**: bar chart with both datasets, RAVDESS green (multimodal wins), MELD red (text wins). Use the QA Pulse brand colors from the PDF style guide for consistency with skakarh.com (dark bg `#0d0f12`, accent blue `#3b82f6`, green `#22c55e`).

**Reply strategy**: be ready to drop the blog link in the comments when people ask "where can I read more?" — LinkedIn deprioritizes posts with outbound links in the body, so the comment-link strategy works around that.

**Cross-post**: same content lightly adapted for the QA Pulse newsletter on skakarh.com. Subject line: "Multimodal AI isn't always better — what RAVDESS and MELD taught me."

**Recommended choice**: **Version A** for the main feed (most detail, best for LinkedIn algorithm preference for "thought leadership" content). Version C as a follow-up if engagement is high — repost a week later with the contrarian framing.
