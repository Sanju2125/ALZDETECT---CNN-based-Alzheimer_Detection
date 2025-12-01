import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import cv2
from io import BytesIO
from PIL import Image
from utils.pdf_generator import generate_pdf_report

st.title("📊 Analysis Report")

# CSS 
st.markdown(
    """
    <style>
    .dataframe { 
        width: 100% !important; 
        text-align: left !important; 
    }
    .dataframe th { 
        text-align: center !important; 
        font-size: 20px !important;
        background-color: #f4f4f4 !important;
    }
    .dataframe td {
        padding: 10px !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Retrieve stored batch results from session state
batch_results = st.session_state.get("batch_results", [])

# Ensure batch_results is a valid list
if not isinstance(batch_results, list):
    batch_results = []

# Retrieve CLASS_DESCRIPTIONS before using it
CLASS_DESCRIPTIONS = st.session_state.get("class_descriptions", {})

# ✅ Append single MRI scan result without overwriting
if "uploaded_image" in st.session_state and "predicted_class" in st.session_state:
    single_analysis = {
        "image_name": "Single MRI Scan",
        "mri_image": st.session_state["uploaded_image"],
        "prediction": st.session_state["predicted_class"],
        "description": CLASS_DESCRIPTIONS.get(st.session_state["predicted_class"], "No description available."),
        "heatmap": st.session_state.get("heatmap"),
    }

    # Prevent duplicate entry
    if single_analysis not in batch_results:
        batch_results.insert(0, single_analysis)  

# ✅ Ensure session state is updated correctly
st.session_state["batch_results"] = batch_results

# ✅ Convert UploadedFile objects to temporary file paths
def ensure_image_file(image_obj):
    if isinstance(image_obj, Image.Image):  # If it's a PIL Image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_obj.convert("RGB").save(temp_file.name, format="JPEG")
        return temp_file.name  # Return file path
    elif isinstance(image_obj, str):  # If it's already a file path
        return image_obj
    return None

# Convert images to file paths
for result in st.session_state["batch_results"]:
    if isinstance(result["mri_image"], st.runtime.uploaded_file_manager.UploadedFile):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")  
        temp_file.write(result["mri_image"].read())  
        temp_file.close()
        result["mri_image"] = temp_file.name  # Store the temp file path instead
    else:
        result["mri_image"] = ensure_image_file(result["mri_image"])

# Ensure there are results to display
if batch_results:
    for idx, result in enumerate(batch_results, start=1):
        st.markdown(f"## 🔍 Result No. {idx}")

        # Create a DataFrame for the table
        df = pd.DataFrame({
            "Title": ["ID", "Predicted Class", "Description"],
            "Content": [result["image_name"], f"{result['prediction']}", result["description"]]
        })

        # Convert DataFrame to HTML with styles
        df_html = df.to_html(index=False, escape=False)
        df_html = df_html.replace('<th>', '<th style="text-align:center;">')

        # Display table
        st.markdown(df_html, unsafe_allow_html=True)

        # ✅ Display MRI Scan & Heatmap only for Single Analysis
        if len(batch_results) == 1:
            st.markdown("### 🧠 MRI Scan")

            # ✅ Ensure image can be properly read
            try:
                if isinstance(result["mri_image"], str):  
                    img = Image.open(result["mri_image"]).convert("RGB")  
                elif isinstance(result["mri_image"], st.runtime.uploaded_file_manager.UploadedFile):
                    img = Image.open(BytesIO(result["mri_image"].getvalue())).convert("RGB")
                else:
                    img = None  

                if img:
                    st.image(img, caption=f"Original MRI {idx}", use_container_width=True)
                else:
                    st.error("⚠ Unable to load MRI image.")
            except Exception as e:
                st.error(f"❌ Error loading MRI image: {e}")

            # ✅ Display Heatmap only for Single Analysis
            st.markdown(f"### 🔥 Grad-CAM Heatmap {idx}")
            heatmap = result.get("heatmap", None)  

            if isinstance(heatmap, np.ndarray) and heatmap.size > 0:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  
                st.image(heatmap, caption=f"Grad-CAM Heatmap {idx}", use_container_width=True)
            else:
                st.write("⚠ No heatmap available.")

            st.write("Heatmap Interpretation: Grad-CAM highlights key brain regions used for classification.")
        
        st.markdown("---")  # Separator

report_path = None  # Initialize report_path

# ✅ Check if only a single MRI scan exists
if len(st.session_state["batch_results"]) == 1:
    report_path = generate_pdf_report(single_result=st.session_state["batch_results"][0])
    button_label = "📥 Download Single Report"
else:
    report_path = generate_pdf_report(batch_results=st.session_state["batch_results"])
    button_label = "📥 Download Full Batch Report"

# ✅ Display Download Button Only if Report is Generated
if report_path:
    with open(report_path, "rb") as file:
        st.download_button(
            label=button_label,
            data=file,
            file_name="MRI_Analysis_Report.pdf",
            mime="application/pdf"
        )
    os.remove(report_path)  # Clean up the temporary file after download
else:
    st.error("❌ Report generation failed. Ensure results exist before generating a report.")
