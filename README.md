**🧠 ALZDETECT---CNN-based-Alzheimer_Detection**
<br>Deep learning-based system for automated Alzheimer’s detection using MRI scans, featuring a CNN classifier, Grad-CAM explainability, and a Streamlit web app for single and batch MRI analysis.</br>

---

**📌 Overview**

ALZDETECT is an end-to-end deep learning pipeline for automated Alzheimer’s Disease classification using MRI brain scans.
It includes:

<ul>
  <li>A custom CNN classifier trained on MRI images</li>
  <li>Four Alzheimer’s categories</li>
  <li>Grad-CAM heatmaps for interpretability</li>
  <li>A full Streamlit Web App</li>
  <li>PDF report generation</li>
  <li>Batch MRI folder analysis</li>
</ul>

**🧬 Classes Predicted**

<ul>
  <li>Mild Dementia (MILD)</li>
  <li>Moderate Dementia (MODERATE)</li>
  <li>Very Mild Dementia (VERY MILD)</li>
  <li>Non-Demented (NORMAL)</li>
</ul>

---


**📘 Dataset**
Kaggle Dataset Used:  
👉 https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset  
> *Note: Due to size constraints, the dataset is not uploaded to this repository.*

---


**🧠 Model Architecture**

 Convolutional Neural Network (CNN) consisting of:

<ul>
  <li>4 convolution blocks</li>
  <li>Batch normalization layers</li>
  <li>Dropout regularization</li>
  <li>Fully connected dense layers</li>
  <li>Softmax output for 4-class classification</li>
</ul>

Performance summary:

<ul>
  <li><b>Accuracy:</b> ~81%</li>
  <li><b>High performance</b> for Moderate Demented cases</li>
  <li><b>Lower recall</b> for Very Mild Demented cases (suggests early-stage detection challenges) </li>
</ul>

---


**🔥 Grad-CAM Explainability**

Grad-CAM heatmaps help visualize:

<ul>
  <li>Which brain regions influenced model decisions</li>
  <li>Structural abnormalities across dementia stages</li>
  <li>The reasoning behind each prediction</li>
</ul>

---


**🌐 Streamlit Web Application**

**✔ Features**

<ul>
  <li><b>Single MRI Prediction</b>: Upload MRI → View predicted class, heatmap, and download PDF report</li>
  <li><b>Batch MRI Analysis</b>: Upload folder → Predictions table, heatmaps, and combined report</li>
  <li><b>Heatmap Info Page</b>: Learn how to interpret Grad-CAM maps</li>
  <li><b>PDF Generator</b>: Auto-generates diagnostic reports</li>
</ul>

---


**🛠️ Installation**

Clone the repository:

```bash
git clone https://github.com/<your-username>/ALZDETECT---CNN-based-Alzheimer_Detection.git
cd ALZDETECT---CNN-based-Alzheimer_Detection

#Install dependencies:
pip install -r requirements.txt
```

---

**▶️ Running the App**
```bash
streamlit run StreamlitApp.py
```
Then open:
👉 http://localhost:8501/

---


**📁 Project Structure**

<ul>
  <li>ALZDETECT/
    <ul>
      <li>README.md</li>
      <li>LICENSE</li>
      <li>requirements.txt</li>
      <li>StreamlitApp.py</li>

   <li>models/
        <ul>
          <li>AlzheimerCnn_model_fixed.keras</li>
        </ul>
      </li>

  <li>notebooks/
        <ul>
          <li>training_notebook.ipynb</li>
          <li>model_evaluation.ipynb</li>
        </ul>
      </li>

  <li>Streamlit/
        <ul>
          <li>pages/
            <ul>
              <li>Analysis_Report.py</li>
              <li>Batch_MRI_Analysis.py</li>
              <li>Heatmap.py</li>
              <li>Heatmap_info.py</li>
            </ul>
          </li>
          <li>utils/
            <ul>
              <li>pdf_generator.py</li>
            </ul>
          </li>
        </ul>
      </li>

  <li>.gitignore</li>
    </ul>
  </li>
</ul>

---

**📜 License**

This project is licensed under the MIT License, allowing academic, research, and commercial reuse with attribution.

---

**👩‍🔬 Author**
<br>Sanjana Sawant</br>
🔗 https://www.linkedin.com/in/sanjana-sawant-9a2991268/



