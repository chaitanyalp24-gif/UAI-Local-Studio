import os
import gc
import io
import time
import random
import tempfile
import urllib.parse
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ============================================================
# PAGE CONFIGURATION & CUSTOM CSS THEME
# ============================================================

st.set_page_config(
    page_title="UAI Web Studio - Free AI Video & Image Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI / UX Custom CSS Stylesheet
CUSTOM_CSS = """
<style>
/* Import Inter / Poppins Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background & Main Container */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #f8fafc;
}

/* Card Container Glassmorphism */
.css-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}

/* Header Gradient */
.hero-title {
    background: linear-gradient(90deg, #a855f7 0%, #ec4899 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
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
    box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.39) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(139, 92, 246, 0.55) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    border: none !important;
    padding: 0px 20px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Ad Banner Card */
.ad-container {
    background: rgba(30, 41, 59, 0.5);
    border: 1px dashed rgba(139, 92, 246, 0.4);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 16px 0;
}
.ad-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# BACKEND & MEMORY MANAGEMENT
# ============================================================

def has_imageio_backend():
    try:
        import imageio
        import imageio_ffmpeg
        return True
    except Exception:
        return False

IMAGEIO_AVAILABLE = has_imageio_backend()

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
        return f"{total:.1f} GB Total | {available:.1f} GB Available"
    except Exception:
        return "System Active"

# Optional Local Diffusers Engine (for offline / local GPU power users)
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

if "ad_click_count" not in st.session_state:
    st.session_state.ad_click_count = 0

# ============================================================
# CLOUD & LOCAL GENERATION ENGINES
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

def generate_video_fast(source_image, motion_type="Cinematic Zoom", duration_sec=3, fps=15, motion_speed=1.0):
    import imageio

    w, h = source_image.size
    total_frames = int(duration_sec * fps)
    frames = []

    for i in range(total_frames):
        progress = i / max(1, total_frames - 1)

        if motion_type == "Cinematic Zoom":
            scale = 1.0 + (progress * 0.20 * motion_speed)
            crop_w, crop_h = int(w / scale), int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            cropped = source_image.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((w, h), Image.Resampling.LANCZOS)

        elif motion_type == "Pan Right":
            max_shift = int(w * 0.12 * motion_speed)
            shift = int(progress * max_shift)
            crop_w = w - max_shift
            cropped = source_image.crop((shift, 0, shift + crop_w, h))
            frame_img = cropped.resize((w, h), Image.Resampling.LANCZOS)

        elif motion_type == "Pan Left":
            max_shift = int(w * 0.12 * motion_speed)
            shift = int((1.0 - progress) * max_shift)
            crop_w = w - max_shift
            cropped = source_image.crop((shift, 0, shift + crop_w, h))
            frame_img = cropped.resize((w, h), Image.Resampling.LANCZOS)

        elif motion_type == "Pulse & Motion":
            scale = 1.0 + (np.sin(progress * np.pi * 2) * 0.05 * motion_speed)
            crop_w, crop_h = int(w / scale), int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            cropped = source_image.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((w, h), Image.Resampling.LANCZOS)

        elif motion_type == "Dramatic Zoom Out":
            scale = 1.25 - (progress * 0.25 * motion_speed)
            crop_w, crop_h = int(w / scale), int(h / scale)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            cropped = source_image.crop((left, top, left + crop_w, top + crop_h))
            frame_img = cropped.resize((w, h), Image.Resampling.LANCZOS)

        else:
            frame_img = source_image

        frames.append(np.array(frame_img))

    output_folder = Path(tempfile.gettempdir()) / "uai_studio"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"generated_video_{int(time.time())}.mp4"

    imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
    return str(output_path)

# ============================================================
# MONETIZATION & AD COMPONENTS
# ============================================================

def render_ad_banner(location="top"):
    """
    Renders customizable ad banner containers compatible with Google AdSense,
    PropellerAds, Ezoic, or custom sponsor banners.
    """
    if location == "top":
        ad_html = """
        <div class="ad-container">
            <div class="ad-label">⚡ Advertisement / Sponsored Banner ⚡</div>
            <div style="background: rgba(139, 92, 246, 0.1); border-radius: 8px; padding: 12px; color: #cbd5e1; font-size: 0.9rem;">
                🚀 <strong>Host & Generate AI Content Free!</strong> Boost your workflow with Unlimited Free Cloud AI Generation.
                <br><span style="font-size: 0.8rem; color: #a7f3d0;">[ Place your Google AdSense / Banner Ad Script Here ]</span>
            </div>
        </div>
        """
        st.markdown(ad_html, unsafe_allow_html=True)

    elif location == "sidebar":
        sidebar_ad_html = """
        <div class="ad-container">
            <div class="ad-label">Sponsor Ads</div>
            <div style="font-size: 0.85rem; color: #94a3b8; padding: 6px;">
                💡 <strong>Support Free AI Hosting</strong><br>
                Clicking ads keeps this generator 100% free for everyone.
            </div>
        </div>
        """
        st.markdown(sidebar_ad_html, unsafe_allow_html=True)

# ============================================================
# HEADER & HERO SECTION
# ============================================================

st.markdown('<h1 class="hero-title">✨ UAI Web Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Instant 100% Free AI Image & Fast Video Generator • Unlimited Usage • Free Cloud Hosting</p>', unsafe_allow_html=True)

# Top Ad Banner Slot
render_ad_banner("top")

st.divider()

# ============================================================
# SIDEBAR CONTROLS & SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Engine & System")

    engine_choice = st.radio(
        "⚡ Generation Engine",
        ["🌐 Fast Free Cloud AI (Recommended - Ultra Fast)", "🖥️ Local CPU Engine (Experimental)"],
        index=0,
        help="Cloud engine generates high resolution images and videos in seconds without local GPU requirements."
    )

    st.info(f"💾 **System RAM:** {get_ram_info()}")

    st.divider()

    st.markdown("### 🎨 Style & Preset Filters")

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

    st.divider()

    st.markdown("### 📐 Image Settings")

    cloud_model = st.selectbox(
        "AI Image Model",
        ["flux", "turbo"],
        index=0,
        help="Flux offers rich details; Turbo offers high speed."
    )

    image_resolution = st.selectbox(
        "Aspect Ratio / Resolution",
        ["512 × 512 (Square)", "512 × 768 (Portrait)", "768 × 512 (Landscape)"],
        index=0
    )

    st.divider()

    st.markdown("### 🎬 Video Motion Settings")

    motion_preset = st.selectbox(
        "Video Motion Mode",
        [
            "Cinematic Zoom",
            "Pan Right",
            "Pan Left",
            "Pulse & Motion",
            "Dramatic Zoom Out"
        ],
        index=0
    )

    video_duration = st.slider("Duration (seconds)", 2, 6, 3)
    video_fps = st.select_slider("Frame Rate (FPS)", options=[10, 15, 24, 30], value=15)

    render_ad_banner("sidebar")

# ============================================================
# MAIN TABS: IMAGE GENERATION & VIDEO CREATION
# ============================================================

image_tab, video_tab, history_tab = st.tabs(
    ["🖼️ AI Image Generator", "🎥 Image-to-Video Generator", "📜 History & Gallery"]
)

# ------------------------------------------------------------
# TAB 1: IMAGE GENERATION
# ------------------------------------------------------------
with image_tab:
    st.markdown("### ✨ Generate Custom AI Artwork")

    col_prompt, col_options = st.columns([3, 1])

    with col_prompt:
        default_prompt = "A majestic futuristic city covered in lush floating gardens, glowing neon waterfalls, ultra detailed, 8k resolution, cinematic lighting"

        user_prompt = st.text_area(
            "Enter Prompt",
            value=default_prompt,
            height=120,
            help="Describe the image you wish to generate."
        )

        # Quick Preset Chips
        st.caption("💡 Quick Prompt Ideas (Click to use):")
        chip_col1, chip_col2, chip_col3 = st.columns(3)
        if chip_col1.button("🌌 Cyberpunk Neon City"):
            user_prompt = "Cyberpunk cityscape at night with neon lights and rain-soaked reflective streets, highly detailed digital painting"
        if chip_col2.button("🐉 Mythical Dragon"):
            user_prompt = "A magnificent golden dragon perched atop a snowy mountain peak, dramatic sunset, fantasy art, photorealistic"
        if chip_col3.button("🚀 Astronaut in Nebula"):
            user_prompt = "An astronaut floating in a colorful cosmos nebula with glowing stars, cosmic cinematic aesthetic"

    with col_options:
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="blurry, distorted, low resolution, bad quality, watermark",
            height=80
        )

        seed_input = st.number_input("Random Seed (0 = Random)", min_value=0, value=0, step=1)

    generate_btn = st.button("✨ Generate AI Image", type="primary", use_container_width=True)

    if generate_btn:
        if not user_prompt.strip():
            st.error("⚠️ Please enter a prompt first.")
        else:
            # Apply Style Presets
            final_prompt = user_prompt.strip()
            if style_preset == "🎬 Cinematic & Photorealistic":
                final_prompt += ", cinematic lighting, 8k resolution, highly detailed, photorealistic, shot on 35mm lens"
            elif style_preset == "⛩️ Anime & Studio Ghibli":
                final_prompt += ", anime style, studio ghibli aesthetic, vibrant colors, beautiful anime artwork"
            elif style_preset == "🌆 Cyberpunk & Synthwave":
                final_prompt += ", cyberpunk style, synthwave neon aesthetic, futuristic glowing elements"
            elif style_preset == "🔮 High Fantasy & Digital Painting":
                final_prompt += ", high fantasy art, intricate digital painting, epic composition, unreal engine 5 render"
            elif style_preset == "🧊 3D Octane Render & Realism":
                final_prompt += ", 3d octane render, smooth lighting, ray tracing, photorealistic textures"
            elif style_preset == "📸 Retro Vintage Photography":
                final_prompt += ", retro vintage photography, grain, film look, aesthetic 1980s photograph"

            res_parts = image_resolution.split("(")[0].strip().split("×")
            w, h = int(res_parts[0].strip()), int(res_parts[1].strip())

            used_seed = seed_input if seed_input > 0 else random.randint(1000, 999999)

            with st.spinner("✨ Generating high resolution AI image in seconds..."):
                try:
                    cleanup_memory()
                    if "Fast Free Cloud" in engine_choice:
                        image = generate_image_cloud(final_prompt, width=w, height=h, model=cloud_model, seed=used_seed)
                    else:
                        local_pipe = load_local_image_pipeline()
                        if local_pipe is not None:
                            res = local_pipe(prompt=final_prompt, num_inference_steps=4, guidance_scale=0.0, width=w, height=h)
                            image = res.images[0]
                        else:
                            st.warning("Local engine fallback to Cloud Engine.")
                            image = generate_image_cloud(final_prompt, width=w, height=h, model=cloud_model, seed=used_seed)

                    st.session_state.generated_image = image
                    st.session_state.generated_video = None
                    st.session_state.generation_history.append({
                        "type": "image",
                        "prompt": final_prompt,
                        "image": image,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.success("✅ Image generated successfully!")
                except Exception as err:
                    st.error(f"Image generation failed: {err}")

# ------------------------------------------------------------
# DISPLAY GENERATED IMAGE
# ------------------------------------------------------------
if st.session_state.generated_image is not None:
    st.divider()
    st.markdown("### 🖼️ Generated Preview & Actions")

    img_col, action_col = st.columns([2, 1])

    with img_col:
        st.image(st.session_state.generated_image, caption="Current Generated Image", use_container_width=True)

    with action_col:
        img = st.session_state.generated_image
        st.markdown(f"**Resolution:** {img.width} × {img.height} px")
        st.markdown(f"**Format:** PNG")

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        st.download_button(
            "💾 Download HD Image (PNG)",
            data=buf.getvalue(),
            file_name="uai_generated_art.png",
            mime="image/png",
            use_container_width=True,
            type="primary"
        )

        st.info("💡 Ready to animate? Switch to the **'Image-to-Video Generator'** tab above to bring this image to life!")

# ------------------------------------------------------------
# TAB 2: IMAGE TO VIDEO GENERATOR
# ------------------------------------------------------------
with video_tab:
    st.markdown("### 🎥 Instant AI Image-to-Video Generator")

    if st.session_state.generated_image is None:
        st.info("ℹ️ Please generate or select an image in the **'AI Image Generator'** tab first.")
    else:
        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            st.markdown("#### Source Image")
            st.image(st.session_state.generated_image, use_container_width=True)

        with v_col2:
            st.markdown("#### Motion Settings")
            st.write(f"**Selected Motion:** {motion_preset}")
            st.write(f"**Target Duration:** {video_duration} Seconds")
            st.write(f"**Frame Rate:** {video_fps} FPS")

            motion_speed = st.slider("Motion Speed Intensity", 0.5, 2.0, 1.0, 0.1)

            generate_vid_btn = st.button("🎬 Animate Image to Video", type="primary", use_container_width=True)

        if generate_vid_btn:
            with st.spinner("🎬 Generating motion video in seconds..."):
                try:
                    video_path = generate_video_fast(
                        st.session_state.generated_image,
                        motion_type=motion_preset,
                        duration_sec=video_duration,
                        fps=video_fps,
                        motion_speed=motion_speed
                    )
                    st.session_state.generated_video = video_path
                    st.session_state.generation_history.append({
                        "type": "video",
                        "path": video_path,
                        "time": time.strftime("%H:%M:%S")
                    })
                    st.success("✅ Video created successfully!")
                except Exception as vid_err:
                    st.error(f"Video generation failed: {vid_err}")

if st.session_state.generated_video:
    st.divider()
    st.markdown("### 🎬 Generated AI Video Output")

    v_preview_col, v_dl_col = st.columns([2, 1])

    with v_preview_col:
        st.video(st.session_state.generated_video)

    with v_dl_col:
        st.markdown("**Video Format:** MP4 (H.264)")
        st.markdown(f"**FPS:** {video_fps}")

        with open(st.session_state.generated_video, "rb") as vf:
            st.download_button(
                "💾 Download Video (MP4)",
                data=vf.read(),
                file_name="uai_ai_video.mp4",
                mime="video/mp4",
                use_container_width=True,
                type="primary"
            )

# ------------------------------------------------------------
# TAB 3: HISTORY & GALLERY
# ------------------------------------------------------------
with history_tab:
    st.markdown("### 📜 Session Generation History")
    if not st.session_state.generation_history:
        st.info("No items generated yet in this session.")
    else:
        for idx, item in enumerate(reversed(st.session_state.generation_history)):
            with st.container():
                st.markdown(f"**Item #{len(st.session_state.generation_history) - idx}** ({item['time']})")
                if item["type"] == "image":
                    st.image(item["image"], width=300)
                    st.caption(item["prompt"])
                elif item["type"] == "video":
                    st.video(item["path"])
                st.divider()

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("UAI Web Studio • 100% Free AI Engine • Ready for Free Cloud Deployment (Streamlit Cloud & Hugging Face Spaces)")
