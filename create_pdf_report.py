
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

TEXT_REPORT = REPORT_DIR / "model_evaluation_report.txt"
PDF_REPORT = REPORT_DIR / "model_evaluation_report.pdf"

if not TEXT_REPORT.exists():
    raise FileNotFoundError(
        f"Text evaluation report not found: {TEXT_REPORT}"
    )

report_text = TEXT_REPORT.read_text(encoding="utf-8")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="MainTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    textColor=colors.HexColor("#163A5F"),
    fontSize=18,
    leading=23,
    spaceAfter=8,
))

styles.add(ParagraphStyle(
    name="Subtitle",
    parent=styles["Normal"],
    alignment=TA_CENTER,
    textColor=colors.HexColor("#45657D"),
    fontSize=10,
    leading=14,
    spaceAfter=16,
))

styles.add(ParagraphStyle(
    name="ReportBody",
    fontName="Courier",
    fontSize=7,
    leading=9,
    textColor=colors.HexColor("#202A33"),
))

doc = SimpleDocTemplate(
    str(PDF_REPORT),
    pagesize=landscape(A4),
    rightMargin=1.5 * cm,
    leftMargin=1.5 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.5 * cm,
)

story = [
    Paragraph(
        "AI-Based Cyber Security Dashboard",
        styles["MainTitle"],
    ),
    Paragraph(
        "Network Intrusion Detection — Model Evaluation Report",
        styles["Subtitle"],
    ),
    Spacer(1, 4),
]

# Keep the complete evaluation report, including both models,
# per-category metrics, confusion matrices, and limitations.
story.append(Preformatted(
    report_text,
    styles["ReportBody"],
    maxLineLength=180,
))

def add_footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        1.5 * cm,
        0.8 * cm,
        "AI Cyber Security Dashboard | Model Evaluation",
    )
    canvas.drawRightString(
        landscape(A4)[0] - 1.5 * cm,
        0.8 * cm,
        f"Page {document.page}",
    )
    canvas.restoreState()

doc.build(
    story,
    onFirstPage=add_footer,
    onLaterPages=add_footer,
)

print(f"PDF report created successfully: {PDF_REPORT}")
print(f"Source report: {TEXT_REPORT}")
print(f"Report size: {PDF_REPORT.stat().st_size:,} bytes")
