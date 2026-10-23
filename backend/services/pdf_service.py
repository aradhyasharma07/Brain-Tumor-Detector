import io
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected",
}


class PdfReportService:
    """
    Generates professional, downloadable PDF clinical-style research reports
    for Brain Tumor MRI classification using ReportLab.
    
    Operates strictly in-memory without saving persistent files to disk,
    and excludes local system paths, environment variables, or private keys.
    """

    @classmethod
    def generate_report(
        cls,
        prediction_result: Dict[str, Any],
        original_image: Image.Image,
        overlay_image: Optional[Image.Image] = None,
        report_id: Optional[str] = None
    ) -> bytes:
        """
        Builds the PDF document in an in-memory buffer and returns the raw bytes.
        """
        if not report_id:
            report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A")
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748B")
        )

        section_heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#475569")
        )

        elements = []

        # --- 1. Header & Metadata ---
        header_table_data = [
            [
                Paragraph("<b>NEUROSCAN AI</b><br/><font size=9 color='#64748B'>Automated Brain Tumor MRI Analysis</font>", title_style),
                Paragraph(
                    f"<b>Report ID:</b> {report_id}<br/>"
                    f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
                    f"<b>Model:</b> ResNet-50 Transfer Learning<br/>"
                    f"<b>Input Resolution:</b> 224 x 224 px",
                    subtitle_style
                )
            ]
        ]
        header_table = Table(header_table_data, colWidths=[310, 222])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

        # --- 2. Primary Classification Result Banner ---
        predicted_class_raw = prediction_result.get("predicted_class", "unknown")
        predicted_display = CLASS_DISPLAY_NAMES.get(predicted_class_raw, predicted_class_raw.capitalize())
        confidence = float(prediction_result.get("confidence", 0.0))
        confidence_percent = f"{confidence * 100:.2f}%"

        # Theme color for banner
        if predicted_class_raw == "notumor":
            banner_bg = colors.HexColor("#ECFDF5")
            banner_border = colors.HexColor("#10B981")
            text_color = colors.HexColor("#065F46")
        else:
            banner_bg = colors.HexColor("#FEF2F2")
            banner_border = colors.HexColor("#EF4444")
            text_color = colors.HexColor("#991B1B")

        primary_result_data = [
            [
                Paragraph("<b>PRIMARY CLASSIFICATION RESULT</b>", ParagraphStyle("PHead", fontName="Helvetica-Bold", fontSize=9, textColor=text_color)),
                Paragraph("<b>CONFIDENCE SCORE</b>", ParagraphStyle("PHead2", fontName="Helvetica-Bold", fontSize=9, textColor=text_color, alignment=2))
            ],
            [
                Paragraph(f"<font size=16><b>{predicted_display}</b></font>", ParagraphStyle("PVal", fontName="Helvetica-Bold", fontSize=16, textColor=text_color)),
                Paragraph(f"<font size=16><b>{confidence_percent}</b></font>", ParagraphStyle("PVal2", fontName="Helvetica-Bold", fontSize=16, textColor=text_color, alignment=2))
            ]
        ]
        primary_table = Table(primary_result_data, colWidths=[340, 192])
        primary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
            ("BOX", (0, 0), (-1, -1), 1, banner_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.transparent),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        elements.append(primary_table)
        elements.append(Spacer(1, 12))

        # --- 3. Complete Class Probabilities Table ---
        elements.append(Paragraph("Diagnostic Class Distribution", section_heading))
        probabilities = prediction_result.get("probabilities", {})

        prob_table_data = [
            [
                Paragraph("<b>Class Name</b>", body_style),
                Paragraph("<b>Clinical Category</b>", body_style),
                Paragraph("<b>Raw Score</b>", body_style),
                Paragraph("<b>Probability</b>", body_style),
                Paragraph("<b>Status</b>", body_style)
            ]
        ]

        canonical_classes = ["glioma", "meningioma", "pituitary", "notumor"]
        for cls_key in canonical_classes:
            prob = float(probabilities.get(cls_key, 0.0))
            is_top = (cls_key == predicted_class_raw)
            status_text = "<b>PREDICTED CLASS</b>" if is_top else "Alternative"
            display_name = CLASS_DISPLAY_NAMES.get(cls_key, cls_key.capitalize())
            category = "Intracranial Neoplasm" if cls_key != "notumor" else "Healthy / No Mass"

            row_style = ParagraphStyle(
                f"Row_{cls_key}",
                parent=body_style,
                fontName="Helvetica-Bold" if is_top else "Helvetica",
                textColor=colors.HexColor("#0284C7") if is_top else colors.HexColor("#334155")
            )

            prob_table_data.append([
                Paragraph(display_name, row_style),
                Paragraph(category, row_style),
                Paragraph(f"{prob:.5f}", row_style),
                Paragraph(f"{prob * 100:.2f}%", row_style),
                Paragraph(status_text, row_style)
            ])

        prob_table = Table(prob_table_data, colWidths=[120, 152, 80, 80, 100])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ALIGN", (2, 0), (3, -1), "RIGHT"),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 14))

        # --- 4. Visual Evidence: Original MRI & Grad-CAM Heatmap ---
        elements.append(Paragraph("Visual Evidence & Explainability (Grad-CAM)", section_heading))

        # Prepare images for embedding
        img_size = (180, 180)

        # Original MRI image buffer
        orig_buf = io.BytesIO()
        original_image.convert("RGB").resize((224, 224)).save(orig_buf, format="JPEG", quality=90)
        orig_buf.seek(0)
        rl_orig_img = RLImage(orig_buf, width=img_size[0], height=img_size[1])

        # Overlay buffer or placeholder
        if overlay_image is not None:
            overlay_buf = io.BytesIO()
            overlay_image.convert("RGB").resize((224, 224)).save(overlay_buf, format="JPEG", quality=90)
            overlay_buf.seek(0)
            rl_overlay_img = RLImage(overlay_buf, width=img_size[0], height=img_size[1])
            overlay_caption = "Grad-CAM Heatmap Overlay (layer4[-1])"
        else:
            rl_overlay_img = Paragraph("<i>Grad-CAM heatmap generation unavailable for this scan.</i>", body_style)
            overlay_caption = "Grad-CAM Unavailable"

        visual_table_data = [
            [
                Paragraph("<b>Input Brain MRI Scan</b>", body_style),
                Paragraph(f"<b>{overlay_caption}</b>", body_style)
            ],
            [
                rl_orig_img,
                rl_overlay_img
            ],
            [
                Paragraph("<font size=8 color='#64748B'>Axial T1/T2-weighted scan slice processed by ResNet-50</font>", body_style),
                Paragraph("<font size=8 color='#64748B'>Warm colors (red/yellow) indicate salient features guiding prediction</font>", body_style)
            ]
        ]
        visual_table = Table(visual_table_data, colWidths=[266, 266])
        visual_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(KeepTogether(visual_table))
        elements.append(Spacer(1, 12))

        # --- 5. Prominent Research & Educational Disclaimer ---
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
        disclaimer_text = (
            "<b>IMPORTANT REGULATORY AND CLINICAL DISCLAIMER:</b><br/>"
            "This report is generated by an automated deep-learning artificial intelligence platform (NeuroScan AI) "
            "strictly for academic research, education, and algorithmic evaluation. This system is <b>NOT</b> an FDA-cleared "
            "medical diagnostic device and has not received regulatory authorization for clinical practice. "
            "The model output and Grad-CAM visualization do not constitute a medical diagnosis, clinical opinion, or "
            "tumor margin boundary specification. Do not use this report for clinical decision making, biopsy planning, "
            "or direct patient treatment. Always obtain independent diagnosis from a board-certified radiologist or physician."
        )
        disclaimer_table = Table(
            [[Paragraph(disclaimer_text, disclaimer_style)]],
            colWidths=[532]
        )
        disclaimer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(KeepTogether(disclaimer_table))

        # Build PDF
        doc.build(elements)
        return buffer.getvalue()
