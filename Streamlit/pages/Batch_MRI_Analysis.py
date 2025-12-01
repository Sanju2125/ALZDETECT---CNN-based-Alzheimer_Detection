import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
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

# Load model only once & store in session_state
model_path = "D:/Documents/Alzheimer/Streamlit/AlzheimerCnn_model_fixed.keras"
if "model" not in st.session_state:
    try:
        st.session_state["model"] = tf.keras.models.load_model(model_path, compile=False)
    except OSError:
        st.error("❌ Error: Unable to load model. Please check the file path or format.")
        st.session_state["model"] = None

# Initialize session variables
for key in ["batch_images", "batch_predictions", "batch_descriptions", "image_names"]:
    if key not in st.session_state:
        st.session_state[key] = {} 

# ✅ Ensure batch_results exists before processing
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []

st.title("📂 Batch MRI Analysis")

# File uploader (max 20 files)
uploaded_files = st.file_uploader(
    "Upload MRI scans (Max: 20)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("⚠ You can upload a maximum of 20 MRI scans.")
    else:
        st.session_state.batch_images = uploaded_files
        st.session_state.image_names = [file.name for file in uploaded_files]

# Function to process and predict images
def preprocess_image(img):
    img = img.convert("RGB")  
    img_array = image.img_to_array(img)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

def predict_image(img):
    model = st.session_state.get("model")
    if model is None:
        return "Model not loaded", "No description available."

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    description = CLASS_DESCRIPTIONS.get(predicted_class, "No description available.")
    return predicted_class, description

# Start Batch Analysis
if st.button("🔍 Start Batch Analysis"):
    if st.session_state.batch_images:
        st.session_state.batch_predictions = []
        st.session_state.batch_descriptions = []

        for file, name in zip(st.session_state.batch_images, st.session_state.image_names):
            img = Image.open(file)
            predicted_class, description = predict_image(img)

            st.session_state.batch_predictions.append(predicted_class)
            st.session_state.batch_descriptions.append(description)

            # ✅ Retrieve heatmap from session (if available)

            # ✅ Append new results with heatmap included
            st.session_state["batch_results"].append({
                "image_name": name,
                "mri_image": file,  
                "prediction": predicted_class,
                "description": description,
            })

        st.success("✅ Batch Analysis Complete!")
# Scrollbar Styling
st.markdown("""
    <style>
        .scrollable-table-container {
            max-width: 100%;
            overflow-x: auto;
            white-space: nowrap;
        }
        .scrollable-table-container table {
            width: 100%;
            display: block;
            overflow-x: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# Display batch results in a Table
if st.session_state.batch_predictions:
    column_names = ["📄 Image Name", "🧠 Predicted Class", "📊 Class Description"]
    results_data = zip(st.session_state.image_names, st.session_state.batch_predictions, st.session_state.batch_descriptions)
    results_df = pd.DataFrame(results_data, columns=column_names)

    # Ensure table only has rows equal to input images
    results_df = results_df.iloc[:len(st.session_state.batch_images)]

    # Display the table inside a scrollable container
    st.markdown('<div class="scrollable-table-container">', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

    # Link to Heatmap Page
st.page_link("pages/Heatmap.py", label="🔬 View Heatmap")