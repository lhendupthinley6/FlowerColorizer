import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


MODEL_PATH = "flower_colorization_model.h5"


@st.cache_resource(show_spinner=False)
def load_colorization_model():
    """Load the trained Keras model and report the expected spatial size."""
    model = tf.keras.models.load_model(MODEL_PATH)
    if not hasattr(model, "input_shape") or len(model.input_shape) != 4:
        raise ValueError("Unexpected model input shape; expected 4D tensor.")

    _, height, width, channels = model.input_shape
    if channels != 1:
        raise ValueError(
            f"Model expects {channels} channels but a single grayscale channel is required."
        )
    if height is None or width is None:
        raise ValueError("Model input height/width are undefined.")

    return model, (int(width), int(height))  # PIL uses (width, height)


def preprocess_image(img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Convert to grayscale, resize, and normalize for the network."""
    grayscale = img.convert("L")
    resized = grayscale.resize(target_size, Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = arr[..., np.newaxis]  # (H, W, 1)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 1)
    return arr, grayscale


def postprocess_prediction(prediction: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    """Convert model output back to a displayable RGB image."""
    if prediction.ndim == 4:
        prediction = prediction[0]

    # Keep only the first three channels in case the model returns extra data.
    prediction = prediction[..., :3]
    prediction = np.clip(prediction, 0.0, 1.0)
    rgb = (prediction * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.resize(output_size, Image.LANCZOS)


def choose_display_width(img: Image.Image) -> int:
    """Pick a display width that adapts to image orientation."""
    w, h = img.size
    if w > h * 1.15:  # landscape
        return 420
    if h > w * 1.15:  # portrait
        return 320
    return 360  # near-square


def main():
    st.set_page_config(page_title="Flower Colorizer", page_icon="🌸", layout="wide")
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
            html, body, [class*="css"]  {
                font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
            }
            body {
                background: radial-gradient(circle at 15% 20%, #f9fff7 0, #eef7ff 32%, #f7fbff 70%);
            }
            .hero {
                padding: 1.4rem 1.6rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #0f766e, #1fb2a6);
                color: #f7fffb;
                box-shadow: 0 16px 50px rgba(7, 67, 60, 0.28);
                position: relative;
                overflow: hidden;
            }
            .hero:before, .hero:after {
                content: "";
                position: absolute;
                border-radius: 50%;
                filter: blur(60px);
                opacity: 0.45;
            }
            .hero:before {
                width: 220px; height: 220px;
                background: #ffffff;
                top: -40px; right: -80px;
            }
            .hero:after {
                width: 160px; height: 160px;
                background: #82ffd8;
                bottom: -60px; left: -20px;
            }
            .hero h1 {
                margin: 0;
                font-size: 1.9rem;
                letter-spacing: -0.01em;
            }
            .hero p {
                margin: 0.3rem 0 0;
                color: rgba(247, 255, 251, 0.88);
            }
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.16);
                color: #f7fffb;
                font-size: 0.87rem;
                margin-right: 0.4rem;
            }
            .card {
                padding: 1rem;
                border-radius: 14px;
                border: 1px solid #e8eef6;
                background: #ffffff;
                box-shadow: 0 8px 28px rgba(15, 118, 110, 0.12);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
                <div>
                    <h1>Flower Colorizer</h1>
                    <p>Upload a grayscale flower photo and let the model repaint it in color.</p>
                </div>
                <div>
                    <span class="pill">128×128 input</span>
                    <span class="pill">Grayscale ➜ RGB</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, target_size = load_colorization_model()
    except Exception as exc:  # pragma: no cover - shown in UI
        st.error(f"Could not load model: {exc}")
        return

    with st.sidebar:
        st.header("Model info")
        st.write(f"Input size: {target_size[0]}×{target_size[1]}")
        st.write("Expected channels: 1 (grayscale)")
        st.caption("Tip: Clear, high-contrast grayscale inputs yield better colors.")

    st.markdown("### Upload")
    with st.container():
        uploaded = st.file_uploader(
            "Upload a grayscale flower image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=False,
        )

    if uploaded is None:
        st.info("Select a grayscale image to begin.")
        return

    original = Image.open(uploaded)
    input_tensor, grayscale = preprocess_image(original, target_size)

    with st.spinner("Colorizing..."):
        try:
            prediction = model.predict(input_tensor, verbose=0)
            colorized = postprocess_prediction(prediction, grayscale.size)
        except Exception as exc:  # pragma: no cover - shown in UI
            st.error(f"Colorization failed: {exc}")
            return

    st.markdown("### Result")
    col1, col2 = st.columns(2, gap="large")
    display_width = choose_display_width(grayscale)
    with col1:
        st.markdown('<div class="card"><h4>Input (grayscale)</h4>', unsafe_allow_html=True)
        st.image(grayscale, width=display_width)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>Colorized output</h4>', unsafe_allow_html=True)
        st.image(colorized, width=display_width)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Tip: For best results, upload clear grayscale photos of flowers. "
        "The model was trained on 128×128 inputs and outputs RGB colors scaled to your original image size."
    )


if __name__ == "__main__":
    main()
