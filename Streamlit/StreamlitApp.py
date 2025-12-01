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


# Repo-relative path we expect the model to be at (on Streamlit Cloud)
MODEL_PATH = os.path.join("Streamlit", "AlzheimerCnn_model_fixed.keras")

# Show debugging info in the UI so we can see exactly what the remote instance sees
st.write("**Remote app working directory:**", os.getcwd())
st.write("**Looking for model at (repo-relative):**", MODEL_PATH)

# list files in the Streamlit/ folder (remote)
try:
    streamlit_dir = "Streamlit"
    files = os.listdir(streamlit_dir) if os.path.exists(streamlit_dir) else []
    st.write(f"**Files in {streamlit_dir}/ :**", files)
    if MODEL_PATH in [os.path.join(streamlit_dir, f) for f in files]:
        st.write("Model filename is present in the directory listing.")
except Exception as e:
    st.write("Could not list Streamlit/ directory:", repr(e))

@st.cache_resource
def try_load_model(path):
    """Try to load and return (model, status_message)."""
    if not os.path.exists(path):
        return None, f"not-found:{path}"
    try:
        m = tf.keras.models.load_model(path, compile=False)
        return m, f"loaded:{path}"
    except Exception as e:
        # return the exception text so the UI can display it (safe for logs)
        return None, f"failed-to-load:{path} -> {e!s}"

# load model (cached)
if "model" not in st.session_state:
    model_obj, status = try_load_model(MODEL_PATH)
    st.session_state["model"] = model_obj
    st.session_state["model_load_status"] = status
else:
    model_obj = st.session_state.get("model")
    status = st.session_state.get("model_load_status", "unknown")

# Show load status in UI
if model_obj is None:
    st.error(f"Model load error: {status}")
else:
    st.success(f"Model loaded successfully from: {MODEL_PATH}")

# Provide a fallback upload widget so you can upload model manually (for quick test)
uploaded = st.file_uploader("If repo model missing/invalid, upload .keras or .h5 here to test", type=["keras", "h5"])
if uploaded is not None:
    tmpfile = f"/tmp/{uploaded.name}"
    with open(tmpfile, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        uploaded_model = tf.keras.models.load_model(tmpfile, compile=False)
        st.session_state["model"] = uploaded_model
        st.session_state["model_load_status"] = f"uploaded-loaded:{uploaded.name}"
        st.success("Uploaded model loaded successfully (temporary for this session).")
    except Exception as e:
        st.error(f"Uploaded model failed to load: {e!s}")
# Give a clear UI message if model is missing or failed
if model_obj is None:
    if model_status.startswith("not-found"):
        st.warning(
            "Model file not found in repo. Expected path:\n"
            f"  {MODEL_PATH}\n\n"
            "Please upload the file to that path in your GitHub repo (Streamlit/AlzheimerCnn_model_fixed.keras) "
            "or use the upload control to test this session."
        )
    else:
        st.error(f"Model load error: {model_status}")

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
