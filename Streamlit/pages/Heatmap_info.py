import streamlit as st

st.title("📖 Understanding Grad-CAM Heatmaps for Alzheimer’s MRI Classification")

st.markdown("""
## 🔍 What is a Grad-CAM Heatmap?
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that helps visualize which regions of an image were most influential in a model’s decision. In Alzheimer’s MRI classification, it highlights the brain areas that contributed most to the diagnosis.

---

## 🧠 How to Analyze Your Heatmap Results

### 1️⃣ **Color Interpretation**
- **🔴 Red/Orange Areas**: These are the regions the model focused on most. In MRI scans, these might highlight areas of atrophy, white matter changes, or hippocampal shrinkage.
- **🟡 Yellow Areas**: Moderately important regions.
- **🔵 Blue Areas**: Less influential or background regions.

### 2️⃣ **Comparing Heatmaps for Different Classifications**
| MRI Classification       | Key Heatmap Insights |
|------------------------|---------------------|
| **NonDemented**       | Minimal red areas, suggesting a normal brain structure. |
| **Very Mild Demented** | Light red areas in hippocampus and temporal lobes, indicating early neurodegeneration. |
| **Mild Demented**     | More intense red areas in the hippocampus, cortex, and ventricles. |
| **Moderate Demented** | Strong red activation in multiple brain regions, showing advanced neurodegeneration. |

### 3️⃣ **Common Patterns in Alzheimer’s Diagnosis**
- A **strong activation** around the **hippocampus** and **temporal lobes** is a sign of potential dementia progression.
- If heatmaps show **diffused or unexpected activation**, the model might be overfitting or misclassifying certain cases.
- Always compare your Grad-CAM results with clinical reports for better validation.

---

## 📚 Research Papers on Grad-CAM for Alzheimer’s MRI Analysis
Below are some key studies where Grad-CAM has been used in Alzheimer’s classification:

1. **Deep Learning in Neuroimaging-Based Diagnosis of Alzheimer’s Disease Using Grad-CAM**  
   📄 [Read Here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7266227/)

2. **Explainable AI in MRI-Based Alzheimer’s Detection Using Grad-CAM**  
   📄 [Read Here](https://www.mdpi.com/2075-4418/15/5/612)

3. **Grad-CAM for Visualizing Deep Learning Models in Alzheimer’s Disease MRI Analysis**  
   📄 [Read Here](https://ieeexplore.ieee.org/abstract/document/10689918)

---

## 🎯 Conclusion
Grad-CAM heatmaps provide crucial insights into **why** a deep learning model classifies an MRI as Alzheimer’s positive or negative. However, they should be used alongside clinical evaluations and not as standalone diagnoses.
""")

# link to go back to the main page
if st.button("🔙 Go to Main App Page"):
    st.switch_page("D:/Documents/Alzheimer/Streamlit/StreamlitApp.py")

