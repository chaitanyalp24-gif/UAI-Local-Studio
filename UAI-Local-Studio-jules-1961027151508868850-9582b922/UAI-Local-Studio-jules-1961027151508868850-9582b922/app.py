import os
import gc
import io
import sys
import time
import random
import tempfile
import traceback
import urllib.parse
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

# ============================================================
# PAGE CONFIGURATION & CUSTOM CSS THEME
# ============================================================

st.set_page_config(
    page_title="UAI Local Studio Pro - Google Omni Video, AI Chat & Code Sandbox",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Glassmorphism & Cyberpunk Neon CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Main Container */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
    color: #f8fafc;
}

/* Card Container Glassmorphism */
.css-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
}

/* Header Gradient */
.hero-title {
    background: linear-gradient(90deg, #c084fc 0%, #f43f5e 40%, #38bdf8 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.2rem;
}

/* Metric Pill Badges */
.metric-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: #38bdf8;
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 12px;
}

/* Custom Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease-in-out !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px 0 rgba(139, 92, 246, 0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px 0 rgba(217, 70, 239, 0.6) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: rgba(15, 23, 42, 0.7);
    padding: 8px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 12px;
    color: #94a3b8;
    font-weight: 600;
    border: none !important;
    padding: 0px 18px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Ad Banner Card */
.ad-container {
    background: rgba(30, 41, 59, 0.45);
    border: 1px dashed rgba(168, 85, 247, 0.4);
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    margin: 14px 0;
}
.ad-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* Chat Message Styling */
.chat-user {
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 14px 14px 2px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #f1f5f9;
}

.chat-assistant {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px 14px 14px 2px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #f8fafc;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# SYSTEM HELPERS & MEMORY MANAGEMENT
# ============================================================

def cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def get_ram_info():
    try:
        import psutil
        mem = psutil.virtual_memory()
        total = mem.total / (1024 ** 3)
        available = mem.available / (1024 ** 3)
        return f"{total:.1f} GB Total | {available:.1f} GB Free"
    except Exception:
        return "System Active"

# Optional Local Diffusers Engine
@st.cache_resource(show_spinner=False)
def load_local_image_pipeline():
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        pipe = pipe.to("cpu")
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        return pipe
    except Exception:
        return None

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

if "generated_video" not in st.session_state:
    st.session_state.generated_video = None

if "generation_history" not in st.session_state:
    st.session_state.generation_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your free AI Assistant & Coding Mentor. How can I help you with code, design, or video creation today?"}
    ]

if "total_generations" not in st.session_state:
    st.session_state.total_generations = 0

if "sandbox_code" not in st.session_state:
    st.session_state.sandbox_code = """# Python Sandbox Practice
def fibonacci(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]

print("Fibonacci Numbers (10):", fibonacci(10))
"""

if "web_sandbox_html" not in st.session_state:
    st.session_state.web_sandbox_html = """<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      background: #0f172a;
      color: #38bdf8;
      font-family: sans-serif;
      text-align: center;
      padding: 40px;
    }
    .btn {
      background: linear-gradient(135deg, #8b5cf6, #d946ef);
      color: white;
      border: none;
      padding: 12px 24px;
      font-size: 16px;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
    }
  </style>
</head>
<body>
  <h1>✨ Welcome to UAI Web Sandbox</h1>
  <p>Edit HTML/CSS/JS and see live results instantly!</p>
  <button class="btn" onclick="alert('Hello from UAI Web Studio!')">Click Me</button>
</body>
</html>"""

# ============================================================
# FREE CLOUD AI ENGINES (IMAGE, OMNI VIDEO & TEXT CHAT)
# ============================================================

def generate_image_cloud(prompt, width=512, height=512, model="flux", seed=None):
    clean_prompt = urllib.parse.quote(prompt.strip())
    seed_param = f"&seed={seed}" if seed is not None else ""
    url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&model={model}&nologo=true{seed_param}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=45)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    else:
        raise Exception(f"Cloud generation API returned status code {response.status_code}")

def generate_video_omni(source_image, motion_type="3D Dolly Zoom", duration_sec=3, fps=15, motion_speed=1.0, target_aspect="16:9"):
    import imageio

    # Calculate target aspect ratio dimensions
    orig_w, orig_h = source_image.size
    if target_aspect == "16:9":
        tw, th = 768, 432
    elif target_aspect == "9:16":
        tw, th = 432, 768
    elif target_aspect == "4:3":
        tw, th = 640, 480
    else:  # 1:1
        tw, th = 512, 512

    base_img = ImageOps.fit(source_image, (tw, th), Image.Resampling.LANCZOS)
    total_frames = int(duration_sec * fps)
    frames = []

    for i in range(total_frames):
        progress = i / max(1, total_frames - 1)

        if motion_type == "3D Dolly Zoom":
            # Reverse perspective scaling
            scale = 1.0 + (progress * 0.28 * motion_speed)
            crop_w, crop_h = int(tw / scale), int(th / scale)
            left = (tw - crop_w) // 2
            top = (th - crop_h) // 2
            cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)
            if progress > 0.5:
                enhancer = ImageEnhance.Contrast(frame_img)
                frame_img = enhancer.enhance(1.0 + (progress - 0.5) * 0.2)

        elif motion_type == "Drone Orbit":
            # Orbital shift with slight tilt
            dx = int(np.sin(progress * np.pi * 2) * 20 * motion_speed)
            dy = int(np.cos(progress * np.pi * 2) * 12 * motion_speed)
            scale = 1.08 + (np.sin(progress * np.pi) * 0.08 * motion_speed)
            crop_w, crop_h = int(tw / scale), int(th / scale)
            left = max(0, min(tw - crop_w, (tw - crop_w) // 2 + dx))
            top = max(0, min(th - crop_h, (th - crop_h) // 2 + dy))
            cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)

        elif motion_type == "Pan Sweep (Left-to-Right)":
            max_shift = int(tw * 0.15 * motion_speed)
            shift = int(progress * max_shift)
            crop_w = tw - max_shift
            cropped = base_img.crop((shift, 0, shift + crop_w, th))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)

        elif motion_type == "Cinematic Zoom In":
            scale = 1.0 + (progress * 0.22 * motion_speed)
            crop_w, crop_h = int(tw / scale), int(th / scale)
            left = (tw - crop_w) // 2
            top = (th - crop_h) // 2
            cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)

        elif motion_type == "Cyber Morph & Pulse":
            scale = 1.0 + (np.sin(progress * np.pi * 3) * 0.06 * motion_speed)
            crop_w, crop_h = int(tw / scale), int(th / scale)
            left = (tw - crop_w) // 2
            top = (th - crop_h) // 2
            cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)

        elif motion_type == "Pulsing Wave":
            scale = 1.15 - (progress * 0.18 * motion_speed)
            crop_w, crop_h = int(tw / scale), int(th / scale)
            left = (tw - crop_w) // 2
            top = (th - crop_h) // 2
            cropped = base_img.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((tw, th), Image.Resampling.LANCZOS)

        else:
            frame_img = base_img

        frames.append(np.array(frame_img))

    output_folder = Path(tempfile.gettempdir()) / "uai_studio"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"omni_video_{int(time.time())}.mp4"

    imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
    return str(output_path)

def call_text_ai(prompt, system_prompt="You are a helpful AI software engineer and creative expert."):
    """
    100% Free AI Chat provider with multi-fallback API structure.
    """
    # Fallback 1: Pollinations POST Json endpoint
    try:
        res = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=20
        )
        if res.status_code == 200 and len(res.text.strip()) > 0:
            return res.text.strip()
    except Exception:
        pass

    # Fallback 2: Pollinations GET encoded endpoint
    try:
        encoded_prompt = urllib.parse.quote(f"{system_prompt}\nUser question: {prompt}")
        res2 = requests.get(f"https://text.pollinations.ai/{encoded_prompt}", timeout=15)
        if res2.status_code == 200 and len(res2.text.strip()) > 0:
            return res2.text.strip()
    except Exception:
        pass

    return "🤖 AI Response: I encountered a brief network delay. Please try resubmitting your question!"

# ============================================================
# MONETIZATION & AD COMPONENTS
# ============================================================

def render_ad_banner(location="top"):
    if location == "top":
        ad_html = """
        <div class="ad-container">
            <div class="ad-label">⚡ Sponsor / Advertisement Banner ⚡</div>
            <div style="background: rgba(139, 92, 246, 0.12); border-radius: 10px; padding: 12px; color: #cbd5e1; font-size: 0.88rem;">
                🚀 <strong>Host & Generate AI Videos 100% Free!</strong> Unlimited Free Cloud Generations & Interactive Code Practice.
                <br><span style="font-size: 0.78rem; color: #38bdf8;">[ Google AdSense / PropellerAds Banner Script Location ]</span>
            </div>
        </div>
        """
        st.markdown(ad_html, unsafe_allow_html=True)

    elif location == "sidebar":
        sidebar_ad_html = """
        <div class="ad-container">
            <div class="ad-label">Sponsor Ads</div>
            <div style="font-size: 0.82rem; color: #94a3b8; padding: 4px;">
                💡 <strong>Support 100% Free AI Hosting</strong><br>
                Sponsored ads keep servers running with zero user fees.
            </div>
        </div>
        """
        st.markdown(sidebar_ad_html, unsafe_allow_html=True)

# ============================================================
# HEADER & HERO SECTION
# ============================================================

st.markdown('<h1 class="hero-title">🤖 UAI Local Studio Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Google Omni-Style Motion AI Video • Free AI Chat & Code Sandbox • 100% Free Server Hosting</p>', unsafe_allow_html=True)

# Metric Badges
st.markdown(f"""
<span class="metric-pill">⚡ Engine: Pollinations Free Cloud</span>
<span class="metric-pill">💾 RAM: {get_ram_info()}</span>
<span class="metric-pill">🔥 Generations: {st.session_state.total_generations}</span>
<span class="metric-pill">💰 Server Cost: $0.00 / Free</span>
""", unsafe_allow_html=True)

render_ad_banner("top")
st.divider()

# ============================================================
# SIDEBAR CONTROLS & SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Engine & Hardware Settings")

    engine_choice = st.radio(
        "⚡ AI Engine",
        ["🌐 Fast Free Cloud AI (Recommended - 0 Cost)", "🖥️ Local CPU Engine"],
        index=0
    )

    st.divider()

    st.markdown("### 🎬 Google Omni Video Controls")

    omni_motion = st.selectbox(
        "Omni Camera Motion",
        [
            "3D Dolly Zoom",
            "Drone Orbit",
            "Cinematic Zoom In",
            "Pan Sweep (Left-to-Right)",
            "Cyber Morph & Pulse",
            "Pulsing Wave"
        ],
        index=0
    )

    omni_aspect = st.selectbox(
        "Video Aspect Ratio",
        ["16:9", "9:16", "1:1", "4:3"],
        index=0
    )

    video_duration = st.slider("Duration (seconds)", 2, 8, 3)
    video_fps = st.select_slider("Frame Rate (FPS)", options=[12, 15, 24, 30], value=15)
    omni_speed = st.slider("Motion Speed", 0.5, 2.5, 1.0, 0.1)

    st.divider()

    st.markdown("### 🎨 Style Filters")
    style_preset = st.selectbox(
        "Artistic Style Filter",
        [
            "None / Custom Prompt",
            "🎬 Cinematic & Photorealistic",
            "⛩️ Anime & Studio Ghibli",
            "🌆 Cyberpunk & Synthwave",
            "🔮 High Fantasy & Digital Painting",
            "🧊 3D Octane Render & Realism",
            "📸 Retro Vintage Photography"
        ]
    )

    render_ad_banner("sidebar")

# ============================================================
# MAIN APPLICATION TABS
# ============================================================

omni_tab, image_tab, chat_code_tab, gallery_tab, deploy_tab = st.tabs([
    "🎥 Google Omni Video",
    "🖼️ AI Image Generator",
    "💻 AI Chat & Code Sandbox",
    "📜 Gallery & History",
    "🚀 100% Free Hosting Guide"
])

# ------------------------------------------------------------
# TAB 1: GOOGLE OMNI VIDEO GENERATOR
# ------------------------------------------------------------
with omni_tab:
    st.markdown("### 🎥 Google Omni-Style Motion AI Video Generator")
    st.caption("Generate cinematic video motion clips with zero GPU server costs.")

    col_src_img, col_omni_opts = st.columns([1, 1])

    with col_src_img:
        st.markdown("#### 1. Select or Upload Image Source")
        if st.session_state.generated_image is not None:
            st.image(st.session_state.generated_image, caption="Current Active AI Image", use_container_width=True)
            use_current = st.checkbox("Use current AI generated image", value=True)
        else:
            use_current = False

        uploaded_file = st.file_uploader("Or upload custom image file (PNG/JPG)", type=["png", "jpg", "jpeg"])

        source_img_to_animate = None
        if uploaded_file is not None:
            source_img_to_animate = Image.open(uploaded_file).convert("RGB")
        elif use_current and st.session_state.generated_image is not None:
            source_img_to_animate = st.session_state.generated_image

    with col_omni_opts:
        st.markdown("#### 2. Omni Camera Vector Configuration")
        st.write(f"**Selected Vector:** {omni_motion}")
        st.write(f"**Aspect Ratio:** {omni_aspect}")
        st.write(f"**Target FPS:** {video_fps} FPS | **Duration:** {video_duration} Seconds")

        animate_omni_btn = st.button("🚀 Render Google Omni Video", type="primary", use_container_width=True)

    if animate_omni_btn:
        if source_img_to_animate is None:
            st.error("⚠️ Please upload an image or generate an AI image first!")
        else:
            with st.spinner("✨ Rendering Google Omni high-speed motion video..."):
                try:
                    v_path = generate_video_omni(
                        source_img_to_animate,
                        motion_type=omni_motion,
                        duration_sec=video_duration,
                        fps=video_fps,
                        motion_speed=omni_speed,
                        target_aspect=omni_aspect
                    )
                    st.session_state.generated_video = v_path
                    st.session_state.total_generations += 1
                    st.session_state.generation_history.append({
                        "type": "video",
                        "path": v_path,
                        "motion": omni_motion,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.success("✅ Google Omni-Style Video Rendered!")
                except Exception as omni_err:
                    st.error(f"Omni video generation failed: {omni_err}")

    if st.session_state.generated_video:
        st.divider()
        st.markdown("### 🎬 Generated Omni Video Output")

        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            st.video(st.session_state.generated_video)
        with v_col2:
            st.markdown(f"**Format:** MP4 (H.264)")
            st.markdown(f"**Camera Vector:** {omni_motion}")
            st.markdown(f"**FPS / Speed:** {video_fps} FPS @ {omni_speed}x")

            with open(st.session_state.generated_video, "rb") as vf:
                st.download_button(
                    "💾 Download Omni Video (MP4)",
                    data=vf.read(),
                    file_name="google_omni_ai_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )

# ------------------------------------------------------------
# TAB 2: AI IMAGE GENERATOR
# ------------------------------------------------------------
with image_tab:
    st.markdown("### 🖼️ Instant Free AI Image Generator")

    col_p, col_o = st.columns([3, 1])

    with col_p:
        default_prompt = "A majestic futuristic sci-fi metropolis with floating glass skyscrapers, glowing cybernetic neon bridges, 8k resolution, cinematic photorealistic lighting"

        user_prompt = st.text_area("Describe your image prompt", value=default_prompt, height=110)

        st.caption("💡 Quick Prompt Ideas (Click to populate):")
        c1, c2, c3 = st.columns(3)
        if c1.button("🌌 Cyberpunk Neon City"):
            user_prompt = "Cyberpunk cityscape at night with neon lights and rain-soaked reflective streets, highly detailed digital painting"
        if c2.button("🐉 Golden Fantasy Dragon"):
            user_prompt = "A magnificent golden dragon perched atop a snowy mountain peak, dramatic sunset, fantasy art, photorealistic"
        if c3.button("🚀 Cosmic Astronaut"):
            user_prompt = "An astronaut floating in a colorful cosmos nebula with glowing stars, cosmic cinematic aesthetic"

    with col_o:
        cloud_model = st.selectbox("Cloud Model", ["flux", "turbo"], index=0)
        res_choice = st.selectbox("Resolution", ["512 × 512 (1:1)", "768 × 512 (16:9)", "512 × 768 (9:16)"], index=0)
        seed_val = st.number_input("Seed (0 = Random)", min_value=0, value=0)

    gen_img_btn = st.button("✨ Generate AI Image", type="primary", use_container_width=True)

    if gen_img_btn:
        if not user_prompt.strip():
            st.error("⚠️ Prompt cannot be empty.")
        else:
            final_p = user_prompt.strip()
            if style_preset == "🎬 Cinematic & Photorealistic":
                final_p += ", cinematic lighting, 8k resolution, photorealistic"
            elif style_preset == "⛩️ Anime & Studio Ghibli":
                final_p += ", anime style, studio ghibli aesthetic"
            elif style_preset == "🌆 Cyberpunk & Synthwave":
                final_p += ", cyberpunk style, neon synthwave aesthetic"
            elif style_preset == "🔮 High Fantasy & Digital Painting":
                final_p += ", high fantasy art, intricate digital painting"

            dim_parts = res_choice.split("(")[0].strip().split("×")
            w, h = int(dim_parts[0].strip()), int(dim_parts[1].strip())
            seed_used = seed_val if seed_val > 0 else random.randint(1000, 999999)

            with st.spinner("✨ Generating AI Image..."):
                try:
                    cleanup_memory()
                    if "Fast Free Cloud" in engine_choice:
                        img = generate_image_cloud(final_p, width=w, height=h, model=cloud_model, seed=seed_used)
                    else:
                        local_p = load_local_image_pipeline()
                        if local_p is not None:
                            res = local_p(prompt=final_p, num_inference_steps=4, guidance_scale=0.0, width=w, height=h)
                            img = res.images[0]
                        else:
                            img = generate_image_cloud(final_p, width=w, height=h, model=cloud_model, seed=seed_used)

                    st.session_state.generated_image = img
                    st.session_state.total_generations += 1
                    st.session_state.generation_history.append({
                        "type": "image",
                        "prompt": final_p,
                        "image": img,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.success("✅ AI Image Created!")
                except Exception as img_err:
                    st.error(f"Image generation failed: {img_err}")

    if st.session_state.generated_image is not None:
        st.divider()
        st.markdown("### 🖼️ Active Generated Image")
        img_preview, img_actions = st.columns([2, 1])

        with img_preview:
            st.image(st.session_state.generated_image, use_container_width=True)

        with img_actions:
            cur_img = st.session_state.generated_image
            st.markdown(f"**Dimensions:** {cur_img.width} × {cur_img.height} px")
            buf = io.BytesIO()
            cur_img.save(buf, format="PNG")

            st.download_button(
                "💾 Download PNG Image",
                data=buf.getvalue(),
                file_name="uai_generated_art.png",
                mime="image/png",
                use_container_width=True,
                type="primary"
            )
            st.info("💡 Tip: Switch to **'Google Omni Video'** tab to animate this image!")

# ------------------------------------------------------------
# TAB 3: AI CHAT ASSISTANT & CODE SANDBOX
# ------------------------------------------------------------
with chat_code_tab:
    st.markdown("### 💻 AI Chat Assistant & Code Sandbox")

    mode_subtab1, mode_subtab2, mode_subtab3 = st.tabs([
        "💬 AI Assistant Chat",
        "🐍 Python Code Sandbox",
        "🌐 Frontend Web Sandbox (HTML/CSS/JS)"
    ])

    # SUBTAB 3.1: AI CHAT ASSISTANT
    with mode_subtab1:
        st.markdown("#### 💬 Multi-Turn AI Coding & Design Assistant")

        # Render Chat History
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><strong>👤 You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant"><strong>🤖 UAI Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

        # Quick Chat Prompt Chips
        st.caption("⚡ Quick Coding Prompts:")
        q1, q2, q3 = st.columns(3)
        prompt_to_send = None

        if q1.button("🐍 Explain Python Asyncio"):
            prompt_to_send = "Explain Python asyncio with a simple code example."
        if q2.button("🎨 CSS Glassmorphism Snippet"):
            prompt_to_send = "Write a modern CSS glassmorphism card style snippet."
        if q3.button("⚡ Fast Fibonacci Generator"):
            prompt_to_send = "Write an optimized Python function for Fibonacci sequence with memoization."

        chat_input = st.text_input("Ask AI Chat Assistant...", key="chat_user_input_field")
        if st.button("Send Message 🚀", type="primary") or prompt_to_send:
            user_q = prompt_to_send if prompt_to_send else chat_input
            if user_q and user_q.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.spinner("🤖 AI Assistant is writing response..."):
                    reply = call_text_ai(user_q)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Chat history cleared. What would you like to build next?"}
            ]
            st.rerun()

    # SUBTAB 3.2: PYTHON CODE SANDBOX
    with mode_subtab2:
        st.markdown("#### 🐍 Interactive Python Code Execution Sandbox")
        st.caption("Write, test, and run Python code right inside your browser.")

        st.caption("📚 Load Preset Code Examples:")
        p_c1, p_c2, p_c3 = st.columns(3)

        if p_c1.button("📊 Data Processing Example"):
            st.session_state.sandbox_code = """# Data Processing Demo
import math

data = [12, 45, 67, 89, 23, 56, 78, 90]
avg = sum(data) / len(data)
std_dev = math.sqrt(sum((x - avg) ** 2 for x in data) / len(data))

print(f"Total Elements: {len(data)}")
print(f"Average: {avg:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
"""
        if p_c2.button("🎨 Pillow Image Draw"):
            st.session_state.sandbox_code = """# Pillow Graphic Generator
from PIL import Image, ImageDraw

img = Image.new('RGB', (400, 200), color=(15, 23, 42))
draw = ImageDraw.Draw(img)
draw.rectangle([20, 20, 380, 180], outline=(139, 92, 246), width=4)
draw.text((100, 90), "Hello from UAI Sandbox!", fill=(248, 250, 252))

print("Image generated successfully in memory!")
print("Width x Height:", img.size)
"""
        if p_c3.button("🧠 Recursive Factorial"):
            st.session_state.sandbox_code = """# Recursive Factorial
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

for num in range(1, 10):
    print(f"Factorial({num}) = {factorial(num)}")
"""

        code_input = st.text_area("Python Editor", value=st.session_state.sandbox_code, height=260)
        st.session_state.sandbox_code = code_input

        if st.button("▶️ Execute Python Code", type="primary", use_container_width=True):
            st.markdown("##### 📤 Execution Output")
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            redirected_output = io.StringIO()
            redirected_error = io.StringIO()
            sys.stdout = redirected_output
            sys.stderr = redirected_error

            try:
                exec_globals = {"st": st, "Image": Image, "np": np}
                exec(code_input, exec_globals)
                out = redirected_output.getvalue()
                err = redirected_error.getvalue()

                if out:
                    st.code(out, language="python")
                if err:
                    st.error(err)
                if not out and not err:
                    st.success("✅ Code executed cleanly with no stdout output.")
            except Exception as ex:
                st.error(f"Execution Error:\n{traceback.format_exc()}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    # SUBTAB 3.3: FRONTEND WEB SANDBOX
    with mode_subtab3:
        st.markdown("#### 🌐 Live HTML/CSS/JS Frontend Sandbox & Preview")
        st.caption("Build custom web UI components with real-time browser preview.")

        html_code = st.text_area("HTML/CSS/JS Code", value=st.session_state.web_sandbox_html, height=220)
        st.session_state.web_sandbox_html = html_code

        st.markdown("##### 👁️ Live Interactive Preview")
        components.html(html_code, height=350, scrolling=True)

# ------------------------------------------------------------
# TAB 4: GALLERY & HISTORY
# ------------------------------------------------------------
with gallery_tab:
    st.markdown("### 📜 Session Output History")

    if not st.session_state.generation_history:
        st.info("No items generated in this session yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.generation_history)):
            with st.container():
                st.markdown(f"**Generation #{len(st.session_state.generation_history) - idx}** ({item['time']})")
                if item["type"] == "image":
                    st.image(item["image"], width=320)
                    st.caption(item["prompt"])
                elif item["type"] == "video":
                    st.video(item["path"])
                    st.caption(f"Vector: {item.get('motion', 'Custom Motion')}")
                st.divider()

# ------------------------------------------------------------
# TAB 5: 100% FREE HOSTING & MONETIZATION GUIDE
# ------------------------------------------------------------
with deploy_tab:
    st.markdown("### 🚀 How to Host & Publish Completely Free (Zero Server Fees)")

    st.markdown("""
    #### 1. Host Free on Streamlit Community Cloud
    1. Push your repository to **GitHub**.
    2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
    3. Click **'New app'**, select your repository, set main path to `app.py`, and click **Deploy**.
    4. Your webapp will be live 24/7 at `https://your-app-name.streamlit.app` completely **FREE**.

    #### 2. Host Free on Hugging Face Spaces
    1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
    2. Choose **Streamlit** as the SDK and set hardware to **Free CPU basic**.
    3. Upload `app.py`, `requirements.txt`, and `.streamlit/config.toml`.
    4. Your app is now live for free worldwide!

    #### 3. Earn Pure Profit with Ads
    - **Google AdSense:** Request approval for your custom domain and embed ad scripts in `render_ad_banner()`.
    - **PropellerAds & Ezoic:** Works out-of-the-box with banner script integration.
    - Because server costs are **$0.00**, 100% of ad impressions and click earnings are **pure profit**!
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("UAI Local Studio Pro • Powered by 100% Free Cloud APIs • Zero Server Costs • All Rights Reserved")
