import os
import gc
import io
import tempfile
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from PIL import Image

from diffusers import AutoPipelineForText2Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

# ============================================================
# UAI LOCAL STUDIO
# ============================================================

st.set_page_config(
    page_title="UAI Local Studio",
    page_icon="🎨",
    layout="wide"
)

# ============================================================
# BACKEND CHECK (imageio / imageio-ffmpeg)
# ============================================================

def has_imageio_backend():
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
        return True
    except Exception:
        return False


IMAGEIO_AVAILABLE = has_imageio_backend()

# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        try:
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

        return f"{total:.1f} GB total | {available:.1f} GB available"

    except Exception:
        return "Unavailable"

# ============================================================
# SESSION STATE
# ============================================================

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

if "generated_video" not in st.session_state:
    st.session_state.generated_video = None

# ============================================================
# IMAGE MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_image_pipeline():

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    pipe = pipe.to("cpu")

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    return pipe

# ============================================================
# VIDEO MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_video_pipeline():

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    pipe = pipe.to("cpu")

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    try:
        pipe.unet.enable_forward_chunking()
    except Exception:
        pass

    return pipe

# ============================================================
# RELEASE MODELS
# ============================================================

def release_image_pipeline():
    try:
        load_image_pipeline.clear()
    except Exception:
        pass

    cleanup_memory()


def release_video_pipeline():
    try:
        load_video_pipeline.clear()
    except Exception:
        pass

    cleanup_memory()

# ============================================================
# HEADER
# ============================================================

st.title("🎨 UAI Local Studio")

st.write(
    "Local AI image generation and experimental image-to-video generation."
)

st.caption(
    "Designed for Windows systems using CPU / Intel integrated graphics."
)

if not IMAGEIO_AVAILABLE:
    st.error(
        "⚠️ `imageio` / `imageio-ffmpeg` not found. Video export will fall "
        "back to the legacy OpenCV backend, which produces noticeably "
        "worse (and sometimes broken) MP4 output. Run this once in your "
        "terminal, then restart the app:\n\n"
        "`py -m pip install imageio imageio-ffmpeg`"
    )

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.write("💾 RAM")
    st.write(get_ram_info())

    st.write("🖥️ Execution")
    st.write("CPU")

    st.write("🎮 Graphics")
    st.write("Integrated GPU")

    st.divider()

    st.header("🖼️ Image Settings")

    image_steps = st.select_slider(
        "Inference Steps",
        options=[1, 2, 3, 4],
        value=4
    )

    image_resolution = st.selectbox(
        "Resolution",
        [
            "512 × 512",
            "512 × 768",
            "768 × 512"
        ],
        index=0
    )

    st.divider()

    st.header("🎥 Video Settings")

    video_resolution = st.selectbox(
        "Resolution (W × H)",
        [
            "256 × 144",
            "320 × 192",
            "384 × 224",
            "448 × 256",
            "512 × 288",
        ],
        index=2,
        help=(
            "SVD-XT was trained at 1024×576. Smaller sizes here run "
            "faster on CPU but look softer/lower quality. 384×224 is a "
            "reasonable balance for a laptop CPU."
        )
    )

    video_frames = st.select_slider(
        "Frames",
        options=[8, 10, 14, 18, 25],
        value=14,
        help="SVD-XT was trained for 25 frames. Fewer frames = shorter, faster clips."
    )

    video_steps = st.select_slider(
        "Inference Steps",
        options=[10, 15, 20, 25, 30],
        value=20,
        help="More steps = sharper, more coherent motion, but slower."
    )

    motion_bucket_id = st.slider(
        "Motion amount",
        min_value=40,
        max_value=200,
        value=127,
        step=1,
        help=(
            "Controls how much movement the video has. Too low = "
            "near-static/frozen video. Too high = warped, unstable "
            "motion. 127 is SVD's default and a good starting point."
        )
    )

    noise_aug_strength = st.slider(
        "Noise augmentation",
        min_value=0.0,
        max_value=0.2,
        value=0.02,
        step=0.01,
        help="Higher values add more variation from the source image but reduce fidelity to it."
    )

    video_fps = st.select_slider(
        "Playback FPS",
        options=[6, 7, 8, 10, 12],
        value=7
    )

    st.warning(
        "On CPU, higher resolution / frames / steps can take several "
        "minutes to tens of minutes. Start small, then scale up once "
        "you know your machine's speed."
    )

# ============================================================
# TABS
# ============================================================

image_tab, video_tab = st.tabs(
    [
        "🖼️ Image Generation",
        "🎥 Image → Video"
    ]
)

# ============================================================
# IMAGE GENERATION
# ============================================================

with image_tab:

    st.header("✨ Generate Image")

    prompt = st.text_area(
        "Prompt",
        value=(
            "A highly detailed cinematic futuristic city at night, "
            "glowing neon lights, realistic architecture, "
            "dramatic atmospheric lighting, beautiful composition, "
            "sharp details, professional digital art"
        ),
        height=130
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        value=(
            "blurry, low quality, distorted, deformed, "
            "bad anatomy, duplicate objects, cropped, "
            "watermark, text, logo"
        ),
        height=90
    )

    st.caption(
        "Note: SDXL-Turbo runs without classifier-free guidance "
        "(guidance scale 0), so the negative prompt has no effect. "
        "It's kept here for UI consistency / future model swaps. "
        "Also keep prompts under ~77 CLIP tokens (~50-60 words) — "
        "anything past that gets silently truncated."
    )

    generate_image = st.button(
        "✨ Generate Image",
        type="primary",
        width="stretch"
    )

    if generate_image:

        if not prompt.strip():

            st.error(
                "Please enter a prompt."
            )

        else:

            width = 512
            height = 512

            if image_resolution == "512 × 768":
                width = 512
                height = 768

            elif image_resolution == "768 × 512":
                width = 768
                height = 512

            try:

                cleanup_memory()

                with st.spinner(
                    "Loading SDXL-Turbo..."
                ):

                    pipe = load_image_pipeline()

                with st.spinner(
                    "Generating image locally..."
                ):

                    result = pipe(
                        prompt=prompt.strip(),
                        negative_prompt=negative_prompt.strip(),
                        num_inference_steps=image_steps,
                        guidance_scale=0.0,
                        width=width,
                        height=height
                    )

                    image = result.images[0]

                st.session_state.generated_image = image
                st.session_state.generated_video = None

                st.success(
                    "✅ Image generated successfully."
                )

            except Exception as error:

                st.error(
                    "Image generation failed."
                )

                st.exception(error)

# ============================================================
# IMAGE PREVIEW
# ============================================================

if st.session_state.generated_image is not None:

    st.divider()

    st.subheader("🖼️ Generated Image")

    preview_col, info_col = st.columns(
        [3, 1]
    )

    with preview_col:

        st.image(
            st.session_state.generated_image,
            caption="UAI Generated Image",
            width="stretch"
        )

    with info_col:

        current_image = st.session_state.generated_image

        st.write(
            f"Width: {current_image.width}px"
        )

        st.write(
            f"Height: {current_image.height}px"
        )

        image_buffer = io.BytesIO()

        current_image.save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            "💾 Download PNG",
            data=image_buffer.getvalue(),
            file_name="uai_generated_image.png",
            mime="image/png",
            width="stretch"
        )

# ============================================================
# VIDEO GENERATION
# ============================================================

with video_tab:

    st.header("🎥 Image → Video")

    if st.session_state.generated_image is None:

        st.info(
            "🖼️ Generate an image first."
        )

    else:

        st.image(
            st.session_state.generated_image,
            caption="Source Image",
            width="stretch"
        )

        st.divider()

        st.warning(
            "Video generation is much more memory- and time-intensive "
            "than image generation."
        )

        generate_video = st.button(
            "🎬 Generate Video",
            type="primary",
            width="stretch"
        )

        if generate_video:

            try:

                # ------------------------------------------------
                # RELEASE IMAGE MODEL
                # ------------------------------------------------

                with st.spinner(
                    "Freeing image model memory..."
                ):

                    release_image_pipeline()

                cleanup_memory()

                # ------------------------------------------------
                # LOAD VIDEO MODEL
                # ------------------------------------------------

                with st.spinner(
                    "Loading video model..."
                ):

                    video_pipe = load_video_pipeline()

                cleanup_memory()

                # ------------------------------------------------
                # PARSE RESOLUTION
                # ------------------------------------------------

                res_w, res_h = (
                    int(v.strip())
                    for v in video_resolution.replace("×", "x").split("x")
                )

                # ------------------------------------------------
                # PREPARE IMAGE
                # ------------------------------------------------

                source_image = (
                    st.session_state.generated_image
                    .convert("RGB")
                )

                source_image = source_image.resize(
                    (res_w, res_h),
                    Image.Resampling.LANCZOS
                )

                cleanup_memory()

                # ------------------------------------------------
                # GENERATE VIDEO
                # ------------------------------------------------

                with st.spinner(
                    f"Generating {video_frames}-frame video on CPU "
                    f"({video_steps} steps)... this can take a while."
                ):

                    generator = torch.Generator(
                        device="cpu"
                    ).manual_seed(42)

                    # Decode more frames per chunk when we can afford
                    # it — this reduces flicker/inconsistency between
                    # decoded chunks. Fall back to 1 if memory is tight.
                    preferred_chunk = min(8, video_frames)

                    try:
                        result = video_pipe(
                            image=source_image,
                            num_frames=video_frames,
                            num_inference_steps=video_steps,
                            decode_chunk_size=preferred_chunk,
                            motion_bucket_id=motion_bucket_id,
                            noise_aug_strength=noise_aug_strength,
                            fps=video_fps,
                            generator=generator
                        )
                    except (MemoryError, RuntimeError):
                        cleanup_memory()
                        result = video_pipe(
                            image=source_image,
                            num_frames=video_frames,
                            num_inference_steps=video_steps,
                            decode_chunk_size=1,
                            motion_bucket_id=motion_bucket_id,
                            noise_aug_strength=noise_aug_strength,
                            fps=video_fps,
                            generator=generator
                        )

                    frames = result.frames[0]

                # ------------------------------------------------
                # SAVE VIDEO
                # ------------------------------------------------

                output_folder = (
                    Path(tempfile.gettempdir())
                    / "uai_local_studio"
                )

                output_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_path = (
                    output_folder
                    / "uai_generated_video.mp4"
                )

                if IMAGEIO_AVAILABLE:
                    export_to_video(
                        frames,
                        str(output_path),
                        fps=video_fps
                    )
                else:
                    st.warning(
                        "Exporting with the legacy OpenCV backend — "
                        "install `imageio` + `imageio-ffmpeg` for "
                        "proper, higher-quality MP4 output."
                    )
                    export_to_video(
                        frames,
                        str(output_path),
                        fps=video_fps
                    )

                st.session_state.generated_video = (
                    str(output_path)
                )

                cleanup_memory()

                st.success(
                    "✅ Video generated successfully."
                )

            except Exception as error:

                st.error(
                    "Video generation failed."
                )

                st.exception(error)

                release_video_pipeline()
                cleanup_memory()

# ============================================================
# VIDEO PREVIEW
# ============================================================

if st.session_state.generated_video:

    video_path = Path(
        st.session_state.generated_video
    )

    if video_path.exists():

        st.divider()

        st.subheader("🎬 Generated Video")

        st.video(
            str(video_path)
        )

        with open(
            video_path,
            "rb"
        ) as video_file:

            st.download_button(
                "💾 Download MP4",
                data=video_file.read(),
                file_name="uai_generated_video.mp4",
                mime="video/mp4",
                width="stretch"
            )

    else:

        st.warning(
            "Previously generated video file is no longer available "
            "(temp folder may have been cleared). Please regenerate."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UAI Local Studio • Local AI inference • No paid AI API required"
)