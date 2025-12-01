import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Ensure the model is loaded
model = st.session_state.get("model")
if model is None:
    st.error("⚠ Model not loaded. Please return to the main page and analyze an image first.")
    st.stop()

st.title("🔬 Grad-CAM Heatmap")

def get_last_conv_layer(model):
    """Identify the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return model.layers[-1].name  # Fallback to last layer if no Conv2D is found

def apply_shiny_effect(heatmap):
    """Apply a shiny glowing effect to the Grad-CAM heatmap."""
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    # Convert to color map (glow effect)
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_TURBO)

    # Blend heatmap with a brighter version of itself
    heatmap_glow = cv2.addWeighted(heatmap_color, 1.2, heatmap_color, 0.5, 0)
    
    return heatmap_glow

def generate_gradcam(img, model, predicted_class):
    """Generate a Grad-CAM heatmap with a shiny glow effect."""
    last_conv_layer_name = get_last_conv_layer(model)
    
    img_array = np.array(img.resize((224, 224)).convert("RGB")) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = np.dot(conv_output, pooled_grads[..., np.newaxis])
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)  # Ensure non-negative values
    heatmap /= (np.max(heatmap) + 1e-8)  # Normalize to [0,1]

    # Resize to original image size
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

    # Apply class-specific intensity adjustments
    intensity_factors = {
        "NonDemented": 0.3, 
        "VeryMildDemented": 0.5, 
        "MildDemented": 0.7, 
        "ModerateDemented": 1.2
    }
    heatmap *= intensity_factors.get(predicted_class, 1.0)  # Default to 1.0 if class not found

    # Apply shiny effect
    heatmap_glow = apply_shiny_effect(heatmap)

    return heatmap_glow

# Single MRI Analysis
if "uploaded_image" in st.session_state and "predicted_class" in st.session_state:
    img = st.session_state["uploaded_image"]
    predicted_class = st.session_state["predicted_class"]
    st.image(img, caption=f"Predicted: {predicted_class}", use_container_width=True)
    
    if "heatmap" not in st.session_state or st.session_state["heatmap"] is None:
        heatmap = generate_gradcam(img, model, predicted_class)
        st.session_state["heatmap"] = heatmap.copy()
    
    img_cv = np.array(img)
    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    overlayed_img = cv2.addWeighted(img_cv, 0.6, st.session_state["heatmap"], 0.4, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    st.image(overlayed_img, caption="✨ Shiny Grad-CAM Heatmap Overlay", use_container_width=True)

# Batch MRI Analysis
if "batch_images" in st.session_state and "batch_predictions" in st.session_state:
    if "heatmaps" not in st.session_state:
        st.session_state["heatmaps"] = {}  # Initialize heatmaps as a dictionary

    for img_file, predicted_class in zip(st.session_state["batch_images"], st.session_state["batch_predictions"]):
        image_name = img_file.name  # Use filename as key
        heatmap = generate_gradcam(Image.open(img_file), model, predicted_class)
        st.session_state["heatmaps"][image_name] = heatmap.copy()  # Save using name as key

    for idx, (img, predicted_class) in enumerate(zip(st.session_state["batch_images"], st.session_state["batch_predictions"])):
        with st.container():
            st.image(img, caption=f"MRI Scan {idx+1}: {predicted_class}", use_container_width=True)
            heatmap = st.session_state["heatmaps"].get(img.name)

            if heatmap is not None:
                img_cv = np.array(Image.open(img))
                if img_cv.shape[-1] == 4:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                overlayed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
                overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
                st.image(overlayed_img, caption=f"✨ Shiny Grad-CAM Heatmap {idx+1}", use_container_width=True)

            if idx < len(st.session_state["batch_images"]) - 1:
                st.markdown("---")

# Navigation buttons
st.markdown("---")
st.subheader("📖 Learn How to Analyze Heatmaps")
if st.button("🔍 View Guide"):
    st.switch_page("D:/Documents/Alzheimer/Streamlit/pages/Heatmap_info.py")
if st.button("🔍 View Analysis Report"):
    st.switch_page("D:/Documents/Alzheimer/Streamlit/pages/Analysis_Report.py")
