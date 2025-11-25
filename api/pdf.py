# api/pdf.py — ultra-light version (under 250 MB guaranteed)
import json
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as RLImage, PageBreak, Spacer
from io import BytesIO

def handler(event, context):
    data = json.loads(event["body"])
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(lambda n, **k: ParagraphStyle(name=n, fontSize=32, alignment=1, textColor=colors.HexColor('#00dbde'), **k), name='TitleBig')
    styles.add(lambda n, **k: ParagraphStyle(name=n, fontSize=16, alignment=1, textColor=colors.HexColor('#fc00ff'), **k), name='Subtitle')
    story = [Spacer(1, 3*inch), Paragraph("PRO FORMA AI", styles['TitleBig']),
             Paragraph("Institutional Investment Memorandum", styles['Subtitle']),
             Paragraph(data.get("date", ""), styles.add(ParagraphStyle(name='Date', fontSize=12, alignment=1))),
             PageBreak(), Paragraph("KEY RETURNS & METRICS", styles["Heading1"])]
    t = Table([[k, v] for k, v in {"Base Equity IRR": data["base_irr"], "Median IRR": data["p50"], "95th IRR": data["p95"],
                                   "Min DSCR": data["min_dscr"], "Equity Multiple": data["equity_multiple"]}.items()],
              colWidths=[4.8*inch, 2.2*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.white), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e1e2e')),
                           ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#00dbde'))]))
    story += [t, PageBreak()]
    for title, b64 in [("EQUITY WATERFALL", data["waterfall_png"]), ("SENSITIVITY ANALYSIS", data["sens_png"]),
                       ("DEBT SERVICE COVERAGE RATIO", data["dscr_png"]), ("MONTE CARLO SIMULATION", data["irr_png"])]:
        story += [Paragraph(title, styles["Heading1"]), RLImage(BytesIO(base64.b64decode(b64)), 7.5*inch, 4.5*inch), PageBreak()]
    story += [Paragraph("FULL CASH FLOW & DSCR SCHEDULE", styles["Heading1"]),
              Table(data["cf_table"], colWidths=[0.7*inch]*9,
                    style=TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.gray), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00dbde'))])),
              PageBreak(), Paragraph("CONFIDENTIAL — PRO FORMA AI", styles["Title"])]
    doc.build(story)
    buffer.seek(0)
    return {"statusCode": 200, "headers": {"Content-Type": "application/pdf"}, "body": buffer.read(), "isBase64Encoded": True}
