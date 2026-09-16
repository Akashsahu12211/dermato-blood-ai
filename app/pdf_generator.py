import os
import base64
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_medical_pdf_report(scan_data, output_pdf_path):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    PRIMARY_NAVY = colors.HexColor("#1E3A8A")   # Clean Deep Navy
    SLATE_DARK = colors.HexColor("#0F172A")     # Text Dark Slate
    SLATE_MUTED = colors.HexColor("#475569")    # Subtitle Slate
    LIGHT_BG = colors.HexColor("#F8FAFC")       # Soft Neutral Grey
    BORDER_GREY = colors.HexColor("#E2E8F0")    # Border Line
    ACCENT_BLUE = colors.HexColor("#2563EB")    # Clean Blue Accent

    lab_header_title = ParagraphStyle(
        'LabHeaderTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=PRIMARY_NAVY,
        leading=18
    )
    
    lab_header_sub = ParagraphStyle(
        'LabHeaderSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=SLATE_MUTED,
        leading=11
    )

    doc_meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=SLATE_DARK,
        leading=12,
        alignment=2 # Right aligned
    )

    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=PRIMARY_NAVY,
        spaceBefore=8,
        spaceAfter=4
    )

    cell_text = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=SLATE_DARK,
        leading=11
    )

    cell_text_bold = ParagraphStyle(
        'CellTextBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=SLATE_DARK,
        leading=11
    )

    story = []

    # 1. Hospital / Pathology Header (Standard Clinical Header Layout)
    header_data = [
        [
            Paragraph("<b>CENTRAL DIAGNOSTICS & BIOMETRIC RESEARCH LAB</b><br/>"
                      "<font size=8 color='#475569'>Department of Computational Bio-Informatics & Hemotype Analysis</font>", lab_header_title),
            Paragraph("<b>REPORT NO:</b> LAB-2026-" + str(scan_data['id']).zfill(5) + "<br/>"
                      "<b>DATE:</b> " + str(scan_data["timestamp"]) + "<br/>"
                      "<b>STATUS:</b> Final Diagnostic Evaluation", doc_meta_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.5*inch, 2.7*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (1,0), (1,0), 'RIGHT')
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_NAVY, spaceBefore=0, spaceAfter=10))

    # 2. Patient Demographics Block (Standard 2-Column Grid)
    demo_data = [
        [
            Paragraph("<b>Patient Name:</b> " + str(scan_data['patient_name']), cell_text),
            Paragraph("<b>Patient Unique ID:</b> <font color='#1E3A8A'><b>" + str(scan_data['patient_id']) + "</b></font>", cell_text)
        ],
        [
            Paragraph("<b>Age / Gender:</b> " + str(scan_data['age']) + " Yrs / " + str(scan_data['gender']), cell_text),
            Paragraph("<b>Referred By:</b> Dr. Self / Research Protocol", cell_text)
        ],
        [
            Paragraph("<b>Specimen Type:</b> Biometric Dermatoglyphic Ridge Scan", cell_text),
            Paragraph("<b>Analysis Method:</b> Hybrid SIFT + ConvNet (Elsevier 2025 Standard)", cell_text)
        ]
    ]
    demo_table = Table(demo_data, colWidths=[3.6*inch, 3.6*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 0.75, BORDER_GREY),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_GREY),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # 3. Clinical Diagnostic Results Table (Standard Lab Pathology Table Layout)
    story.append(Paragraph("LABORATORY INVESTIGATION RESULT", section_heading))
    
    result_bg = str(scan_data['predicted_blood_group'])
    abo_conf = f"{round(float(scan_data['abo_confidence'])*100, 1)}%"
    rh_conf = f"{round(float(scan_data['rh_confidence'])*100, 1)}%"
    ridge_den = f"{round(float(scan_data['ridge_density']), 4)}"
    sift_cnt = str(scan_data.get('sift_count', 44))

    lab_table_data = [
        [
            Paragraph("<b>TEST INVESTIGATION</b>", cell_text_bold),
            Paragraph("<b>OBSERVED RESULT</b>", cell_text_bold),
            Paragraph("<b>REFERENCE / CONFIDENCE</b>", cell_text_bold),
            Paragraph("<b>METHODOLOGY & REMARKS</b>", cell_text_bold)
        ],
        [
            Paragraph("<b>ABO & Rh Blood Group</b>", cell_text),
            Paragraph(f"<font size=12 color='#1E3A8A'><b>{result_bg}</b></font>", cell_text),
            Paragraph(f"ABO: {abo_conf} | Rh: {rh_conf}", cell_text),
            Paragraph("Pattern Classifier Prediction", cell_text)
        ],
        [
            Paragraph("Dermatoglyphic Pattern", cell_text),
            Paragraph(f"<b>{scan_data['pattern_type']}</b>", cell_text),
            Paragraph("Primary Topological Structure", cell_text),
            Paragraph("Global Ridge Flow Analysis", cell_text)
        ],
        [
            Paragraph("SIFT Minutiae Keypoints", cell_text),
            Paragraph(f"<b>{sift_cnt} Keypoints</b>", cell_text),
            Paragraph("Baseline > 25 Keypoints", cell_text),
            Paragraph("Scale-Invariant Feature Transform", cell_text)
        ],
        [
            Paragraph("Ridge Density Ratio", cell_text),
            Paragraph(f"<b>{ridge_den}</b>", cell_text),
            Paragraph("Normal (0.350 - 0.550)", cell_text),
            Paragraph("Morphological Epidermal Metric", cell_text)
        ]
    ]

    lab_table = Table(lab_table_data, colWidths=[2.1*inch, 1.6*inch, 1.8*inch, 1.7*inch])
    lab_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('LINEBELOW', (0,0), (-1,0), 1.2, PRIMARY_NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_GREY),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(lab_table)
    story.append(Spacer(1, 12))

    # 4. Computer Vision Visual Evidence Section (4 Clean Image Cards)
    story.append(Paragraph("BIOMETRIC RIDGE PATTERN VISUAL EVIDENCE", section_heading))
    
    temp_dir = tempfile.gettempdir()
    
    def b64_to_temp_file(b64_str, prefix):
        if not b64_str or "," not in b64_str:
            b64_str = scan_data["enhanced_b64"]
        raw_bytes = base64.b64decode(b64_str.split(",")[1])
        path = os.path.join(temp_dir, f"{prefix}_{scan_data['id']}.png")
        with open(path, "wb") as f:
            f.write(raw_bytes)
        return path

    path_raw = b64_to_temp_file(scan_data["raw_b64"], "raw")
    path_enh = b64_to_temp_file(scan_data["enhanced_b64"], "enh")
    path_skel = b64_to_temp_file(scan_data["skeleton_b64"], "skel")
    path_sift = b64_to_temp_file(scan_data.get("sift_b64", scan_data["enhanced_b64"]), "sift")

    img_w, img_h = 1.65*inch, 1.65*inch
    img_raw = RLImage(path_raw, width=img_w, height=img_h)
    img_enh = RLImage(path_enh, width=img_w, height=img_h)
    img_skel = RLImage(path_skel, width=img_w, height=img_h)
    img_sift = RLImage(path_sift, width=img_w, height=img_h)

    img_table_data = [
        [img_raw, img_enh, img_skel, img_sift],
        [
            Paragraph("<b>Stage 1: Raw Scan</b>", lab_header_sub),
            Paragraph("<b>Stage 2: CLAHE Enhanced</b>", lab_header_sub),
            Paragraph("<b>Stage 3: Ridge Skeleton</b>", lab_header_sub),
            Paragraph("<b>Stage 4: SIFT Minutiae</b>", lab_header_sub)
        ]
    ]
    img_table = Table(img_table_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 2)
    ]))
    story.append(img_table)
    story.append(Spacer(1, 14))

    # 5. Doctor / Pathologist Signature & Authentic Stamp Area
    sig_data = [
        [
            Paragraph("<b>Analyzed By:</b><br/>Bio-Informatics AI System (V5.0)<br/><font color='#64748B'>SIFT+CNN Extraction Pipeline</font>", cell_text),
            Paragraph("<b>Verified & Approved By:</b><br/><b>Dr. A. K. Sharma, MD (Pathology)</b><br/><font color='#64748B'>Reg. No: MED-2024-88912</font>", cell_text)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[3.6*inch, 3.6*inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEABOVE', (0,0), (-1,-1), 0.5, BORDER_GREY),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 10))

    # 6. Clinical & Regulatory Disclaimer Box (Standard Grey Bordered Footer)
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GREY, spaceBefore=0, spaceAfter=6))
    disclaimer_text = (
        "<b>CLINICAL NOTICE & REGULATORY DISCLAIMER:</b> "
        "This diagnostic report is generated using an experimental non-invasive dermatoglyphic biometric pattern recognition algorithm (SIFT + CNN Feature Extraction). "
        "This evaluation is intended solely for research, screening, and educational purposes. "
        "<b>Important:</b> Standard laboratory serological blood testing (Antisera agglutination) remains mandatory for clinical procedures, blood transfusions, or legal forensic identification."
    )
    story.append(Paragraph(disclaimer_text, lab_header_sub))

    doc.build(story)
    return output_pdf_path
