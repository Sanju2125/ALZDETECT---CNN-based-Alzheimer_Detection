from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer, Paragraph, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os
import datetime
import numpy as np
import tempfile
from PIL import Image as PILImage

def generate_pdf_report(single_result=None, batch_results=None):  
    report_path = "MRI_Analysis_Report.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # ✅ Add Date and Time
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"<b>Report Generated:</b> {now}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    def create_table(data):
        """Helper function to create tables with styling."""
        table = Table(data, colWidths=[150, 350], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (1, 0), (-1, -1), True),
            ('LEFTPADDING', (1, 0), (-1, -1), 5),
            ('RIGHTPADDING', (1, 0), (-1, -1), 5),
        ]))
        return table

    # ✅ Process Single Analysis Report (With MRI scan and Heatmap)
    if single_result:
        elements.append(Paragraph("<b>Single MRI Analysis Report</b>", styles["Title"]))
        elements.append(Spacer(1, 12))

        data = [
            ["Title", "Content"],
            ["ID", single_result["image_name"]],
            ["Predicted Class", single_result["prediction"]],
            ["Description", Paragraph(single_result["description"], styles["BodyText"])],
        ]
        elements.append(create_table(data))
        elements.append(Spacer(1, 25))  

        # ✅ Add MRI Image
        if os.path.exists(single_result["mri_image"]):
            elements.append(Paragraph("<b>MRI Scan</b>", styles["Heading2"]))
            elements.append(Spacer(1, 10))
            elements.append(Image(single_result["mri_image"], width=300, height=300))
            elements.append(Spacer(1, 20))

        # ✅ Page Break before Grad-CAM Heatmap
        elements.append(PageBreak())

        # ✅ Add Heatmap (Handles NumPy Array & Image Paths)
        heatmap_path = single_result.get("heatmap")  

        if isinstance(heatmap_path, np.ndarray):  # If heatmap is a NumPy array
            # Save the heatmap as a temporary file to avoid distortion
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                heatmap_image = PILImage.fromarray(heatmap_path.copy())  # Use .copy() to prevent distortion
                heatmap_image.save(temp_file, format="PNG")
                heatmap_path = temp_file.name  # Update heatmap_path to the saved file path

        if heatmap_path and os.path.exists(heatmap_path):
            elements.append(Paragraph("<b>Grad-CAM Heatmap</b>", styles["Heading2"]))
            elements.append(Spacer(1, 5))
            elements.append(Image(heatmap_path, width=300, height=300, hAlign='CENTER'))  # Center the heatmap
            elements.append(Spacer(1, 20))

    # ✅ Process Batch Analysis Report (No MRI scans, No Heatmaps)
    if batch_results:
        elements.append(Paragraph("<b>Batch MRI Analysis Report</b>", styles["Title"]))
        elements.append(Spacer(1, 12))

        for idx, result in enumerate(batch_results, start=1):
            elements.append(Paragraph(f"<b>Result {idx}</b>", styles["Heading2"]))
            elements.append(Spacer(1, 8))

            data = [
                ["Title", "Content"],
                ["ID", result["image_name"]],
                ["Predicted Class", result["prediction"]],
                ["Description", Paragraph(result["description"], styles["BodyText"])],
            ]
            elements.append(create_table(data))
            elements.append(Spacer(1, 20))

    # ✅ Finalize and Save PDF
    doc.build(elements)

    # ✅ Ensure File is Fully Written Before Returning Path
    if os.path.exists(report_path):
        return report_path
    else:
        return None  # If file creation failed
