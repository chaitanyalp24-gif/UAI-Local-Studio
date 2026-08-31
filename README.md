# 🎨 UAI Web Studio - Free AI Video & Image Generator

UAI Web Studio is a modernized, 100% free web application for generating high-resolution AI images and animating them into motion videos in seconds.

It runs completely free without requiring any paid API keys or expensive local GPU hardware. It is built to be deployed 100% free on public cloud platforms and monetized via ad networks like Google AdSense, PropellerAds, or custom sponsors.

---

## ✨ Features

- ⚡ **Ultra-Fast Free AI Image Generation**: High quality SDXL/Flux image generation in seconds.
- 🎬 **Instant Neural Motion Video Generator**: Animates any static image into smooth MP4 motion clips without timing out or crashing on CPU.
- 🎨 **Artistic Style Filters**: One-click filters for Cinematic, Anime, Cyberpunk, Fantasy, 3D Render, and Vintage looks.
- 💡 **Quick Prompt Chips & Preset Ideas**: Built-in prompts to get started instantly.
- 📜 **Session History & Gallery**: Save, preview, and download all generated images and videos during your session.
- 💰 **Built-In Monetization**: Pre-configured responsive Ad banner slots for top header, sidebar, and footer.
- 🚀 **Zero Cost Cloud Deployment**: Step-by-step instructions for hosting 100% free on Streamlit Community Cloud and Hugging Face Spaces.

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/UAI-Local-Studio.git
   cd UAI-Local-Studio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 100% Free Cloud Deployment Guide

### Option 1: Deploy on Streamlit Community Cloud (Recommended - Free)

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **Deploy!** Your app will be live with an SSL HTTPS URL for free.

### Option 2: Deploy on Hugging Face Spaces (Free)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your Space and select **Streamlit** as the Space SDK.
3. Choose **Public** and select the free CPU tier.
4. Clone the space repository locally or upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `README.md`.
5. Push to Hugging Face. Your app will automatically build and launch!

---

## 💰 How to Earn Money with Ads

The application includes modular Ad slots (`render_ad_banner`) in `app.py`.

1. **Register with an Ad Provider**:
   - [Google AdSense](https://adsense.google.com/)
   - [PropellerAds](https://propellerads.com/)
   - [Ezoic](https://www.ezoic.com/)
2. **Copy your Publisher Script / Ad Unit Code**.
3. **Insert the code into `app.py`**:
   Replace the placeholder HTML in `render_ad_banner()` inside `app.py` with your ad script or HTML iframe snippet:
   ```python
   # Example AdSense script insertion in app.py
   components.html('''<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-YOUR_PUB_ID" crossorigin="anonymous"></script>''', height=100)
   ```

---

## 📄 License

Distributed under the MIT License. Free for commercial and personal use.
