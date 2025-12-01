import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define class labels and descriptions
CLASS_LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_DESCRIPTIONS = {
    "MildDemented": (
        "MRI analysis indicates mild atrophy in the hippocampus and medial temporal lobes, "
        "which are early markers of neurodegeneration. Mild ventricular enlargement may also be present, "
        "suggesting slight brain volume loss associated with early-stage dementia."
    ),
    "ModerateDemented": (
        "The scan reveals significant shrinkage in the hippocampus and cerebral cortex, "
        "with noticeable ventricular enlargement. White matter hyperintensities may be more pronounced, "
        "indicating moderate progression of neurodegeneration."
    ),
    "NonDemented": (
        "No significant cortical atrophy or ventricular enlargement is observed. "
        "The hippocampal region appears intact, with normal white and gray matter distribution. "
        "Brain volume and structure are consistent with a healthy, non-demented individual."
    ),
    "VeryMildDemented": (
        "Subtle reduction in hippocampal volume and mild cortical thinning are detected, "
        "which may indicate early-stage neurodegeneration. However, these changes are minimal and do not yet "
        "significantly impact cognitive function."
    )
}

# Store class descriptions in session state
if "class_descriptions" not in st.session_state:
    st.session_state["class_descriptions"] = CLASS_DESCRIPTIONS

# Load model only once & store in session_state
MODEL_PATH = os.path.join("Streamlit", "AlzheimerCnn_model_fixed.keras")

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}"

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, "loaded"
    except Exception as e:
        return None, f"Failed to load model: {e}"

model, model_load_status = load_cnn_model()

if model is None:
    st.error(f"❌ {model_load_status}")

def preprocess_image(img):
    """Preprocess the image for CNN prediction."""
    img = img.convert("RGB")  # Ensure correct format
    img_array = image.img_to_array(img)
    img_array = tf.image.resize(img_array, (224, 224))  # Resize efficiently
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

def predict_image(img):
    """Predict the class of an MRI scan using the CNN model."""
    model = st.session_state.get("model")  # Retrieve model from session_state
    if model is None:
        return "Model not loaded", None

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    description = st.session_state["class_descriptions"].get(predicted_class, "No description available.")  # Use session state
    return predicted_class, description

# Streamlit App
st.title("🧠 AlzDetect - Alzheimer MRI Analysis")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["MRI Scan Analysis"])

# MRI SCAN ANALYSIS PAGE
if page == "MRI Scan Analysis":
    st.header("Upload & Analyze MRI Scan")
    mri_file = st.file_uploader("Upload an MRI scan (JPG/PNG)", type=["jpg", "png"])
    
    if mri_file is not None:
        img = Image.open(mri_file)
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

        if st.button("🔍 Analyze MRI Scan"):
            predicted_class, description = predict_image(img)

            if predicted_class == "Model not loaded":
                st.error("❌ The model is not available. Please check the file path.")
            else:
                st.success(f"**Predicted Diagnosis:** {predicted_class}")
                st.write("📊 **MRI Scan Analysis:**")
                st.info(description)
                
                # Save image, predicted class & description to session_state for use in other pages
                st.session_state["uploaded_image"] = img
                st.session_state["predicted_class"] = predicted_class
                st.session_state["predicted_description"] = description
                
                # Link to Heatmap Page
                st.page_link("pages/Heatmap.py", label="🔬 View Heatmap")
