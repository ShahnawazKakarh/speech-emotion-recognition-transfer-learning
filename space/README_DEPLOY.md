# HuggingFace Spaces Deployment Guide

This directory contains a ready-to-deploy Gradio app for HuggingFace Spaces. It's a **standalone** scaffold — the HF Space is a separate git repo on huggingface.co, distinct from this GitHub repo.

The default app uses a public pretrained SER model (`superb/wav2vec2-base-superb-er`) so the Space boots in under a minute with no checkpoint upload required. You can swap in your custom-trained RoBERTa + wav2vec2 cross-attention checkpoint later — see the bottom of this doc.

## 🚀 First-time deploy (15 min, one-time setup)

### 1. Create the Space on HuggingFace

1. Sign up / log in at <https://huggingface.co>
2. Go to <https://huggingface.co/new-space>
3. Fill in:
   - **Owner**: your HF username (e.g. `skakarh`)
   - **Space name**: `speech-emotion-recognition` (or `ser-transfer-learning`)
   - **License**: MIT
   - **Space SDK**: Gradio
   - **Gradio template**: Blank
   - **Hardware**: CPU basic (free tier — plenty for this model)
   - **Visibility**: Public
4. Click **Create Space**.

You'll land on `https://huggingface.co/spaces/<your-username>/speech-emotion-recognition`. It will show "No app file" — that's expected, we haven't pushed anything yet.

### 2. Clone the (empty) Space repo

```bash
# Outside this GitHub repo — pick a parallel folder
cd ~
git clone https://huggingface.co/spaces/<your-username>/speech-emotion-recognition
cd speech-emotion-recognition
```

### 3. Copy the Space files from this GitHub repo

```bash
# From inside the cloned HF Space repo:
cp ~/speech-emotion-recognition-transfer-learning/space/README.md .
cp ~/speech-emotion-recognition-transfer-learning/space/app.py .
cp ~/speech-emotion-recognition-transfer-learning/space/requirements.txt .
```

### 4. Push to HuggingFace

```bash
git add README.md app.py requirements.txt
git commit -m "Initial deploy: pretrained wav2vec2 SER demo"
git push
```

The first push will prompt for your HF credentials. Use a **write token** from <https://huggingface.co/settings/tokens> as the password (not your HF account password).

### 5. Wait for the build

Go back to your Space URL. You'll see the build logs streaming. Should take 2-5 minutes — Spaces install your `requirements.txt` and start the Gradio app. When you see "Running on local URL: http://0.0.0.0:7860", the Space is live.

### 6. Test it

Open the Space URL in your browser, upload a `.wav`, click Predict.

## 🔄 Updating the Space later

Just edit files in your local cloned Space repo and `git push`. The Space rebuilds automatically.

If you change anything in `space/` here in the GitHub repo, copy it over to the cloned Space repo and push.

## 🎯 Upgrading to a custom-trained checkpoint

The current Space uses the public `superb/wav2vec2-base-superb-er` model so it works out-of-the-box. To use your own trained checkpoint (e.g. `outputs/multimodal_ravdess_si/best-08-0.7703.ckpt`):

### Option A: upload checkpoint to HF Model Hub (recommended)

1. Create a HF Model repo: <https://huggingface.co/new>
   - Owner: your HF username
   - Model name: `ser-multimodal-ravdess`
   - License: MIT
2. Clone it: `git clone https://huggingface.co/<your-username>/ser-multimodal-ravdess`
3. Copy your `.ckpt` into it (git-lfs handles the big file automatically on HF)
4. Push.
5. Update the Space's `app.py` to download the checkpoint at startup using `huggingface_hub.hf_hub_download`, load it into `SERLightningModule`, and route inference through your model code (you'll need to also copy `src/models/`, `src/data/ravdess.py`, etc. into the Space repo, OR `pip install` from your GitHub repo via a `requirements.txt` entry like `git+https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning.git`).

### Option B: keep the simple pretrained Space + add a "Compare" tab

Cleaner UX — show predictions from BOTH the public baseline and your custom model side-by-side. Same approach as Option A for the custom model, just with a second prediction call rendered in a separate Gradio Column or Tab.

## 📝 What's deployed by default

- **Model**: `superb/wav2vec2-base-superb-er` (4-class: neu/hap/sad/ang)
- **Inputs**: upload .wav/.mp3/.flac or record from mic
- **Outputs**: top-5 probability bars
- **Hardware**: CPU basic (free, ~1-2s inference latency)

## 🐛 Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `librosa` import | Add `libsndfile1` to a `packages.txt` file in the Space (system deps); typically not needed on HF base images |
| Space stuck on "Building" >10 min | Restart from the Space settings (gear icon) |
| Predict button does nothing | Check the Space logs tab; usually a `transformers` import error |
| "No app file" persists after push | Make sure the file is named exactly `app.py` (lowercase, no underscore) at the Space repo root |
| Push rejected: `large files detected` | You probably committed a `.ckpt`. The Space repo isn't the right place for big binaries — use HF Model Hub (Option A above) |

## 📎 Useful links

- Your Space URL: `https://huggingface.co/spaces/<your-username>/speech-emotion-recognition`
- HF Spaces docs: <https://huggingface.co/docs/hub/spaces>
- Gradio docs: <https://www.gradio.app/docs>
- HF Model Hub (for hosting checkpoints): <https://huggingface.co/new>
