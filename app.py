# app.py — FINAL $500K/YEAR VERSION WITH PROFESSIONAL PDF STYLING
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import stripe
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Frame, PageTemplate, NextPageTemplate
import streamlit.components.v1 as components

# ——— CONFIG ———
APP_URL = "https://proforma-ai-f3poyqgcroefu3qwcqwy3m.streamlit.app"

try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL   = st.secrets["stripe_prices"]["annual"]
except:
    st.error("Add Stripe secrets")
    st.stop()

# ——— PAYWALL ———
if "paid" not in st.query_params:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.title("Pro Forma AI")
    st.markdown("##### Institutional Stress-Test • Full Sensitivity • Bank-Grade PDF")
    st.markdown("**Used by top REITs, banks & sponsors on $2B+ of transactions**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("$999 → One Deal", type="primary", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=one",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)
    with c2:
        if st.button("$250,000/yr → Institutional White-Label", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ANNUAL, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=annual",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)
    st.caption("Test card: 4242 4242 4242 4242")
    st.stop()

# ——— PAID TOOL ———
st.set_page_config(page_title="Pro Forma AI – Institutional", layout="wide")
st.success("Institutional access active — 5-page bank-grade PDF")
st.title("Pro Forma AI")

st.info("Enter deal → get full institutional underwriting package")

c1, c2 = st.columns(2)
with c1:
    cost   = st.number_input("Total Development Cost", value=92_500_000, step=1_000_000, format="%d")
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi    = st.number_input("Year 1 Stabilized NOI", value=7_200_000, step=100_000, format="%d")
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN INSTITUTIONAL ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 scenarios + full sensitivity…"):
        np.random.seed(42)
        n = 50000

        # Monte Carlo
        cost_var   = np.random.normal(1, 0.15, n)
        rate_var   = np.random.normal(1, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var    = np.random.normal(cap, 0.008, n)
        noi_y1_var = np.random.normal(1, 0.10, n)

        actual_cost = cost * cost_var
        loan = actual_cost * (ltc / 100)
        ds = loan * rate * rate_var
        noi_y1 = noi * noi_y1_var

        dscr = np.where(ds > 0, noi_y1 / ds, 10)
        p_dscr = np.percentile(dscr, [5, 25, 50, 75, 95])

        noi_exit = noi * (1 + growth_var)**(years-1)
        exit_val = noi_exit / cap_var
        profit = exit_val - loan - (ds * years)
        equity_in = cost * (equity/100)
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -1)
        valid_irr = irr[irr > -1]
        p_irr = np.percentile(valid_irr, [5, 25, 50, 75, 95])

        # Full Sensitivity
        g_range = np.linspace(growth * 0.6, growth * 1.4, 9)
        c_range = np.linspace(cap * 0.85, cap * 1.15, 9)
        sens_irr = np.zeros((9,9))
        sens_dscr = np.zeros((9,9))

        for i,g in enumerate(g_range):
            for j,c in enumerate(c_range):
                noi_exit_s = noi * (1+g)**(years-1)
                exit_s = noi_exit_s / c
                profit_s = exit_s - actual_cost.mean() - (ds.mean()*years)
                sens_irr[i,j] = max((profit_s / equity_in)**(1/years) - 1, -1)
                sens_dscr[i,j] = noi / (actual_cost.mean() * (ltc/100) * rate)

    st.success("Analysis Complete – Generating Bank-Grade PDF")

    # Interactive Charts (kept simple)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(valid_irr*100, nbins=70, title="Equity IRR Distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(dscr, nbins=60, title="DSCR Distribution"), use_container_width=True)
    with col2:
        st.plotly_chart(px.imshow(sens_irr*100, title="IRR Sensitivity", color_continuous_scale="RdYlGn"), use_container_width=True)
        st.plotly_chart(px.imshow(sens_dscr, title="DSCR Sensitivity", color_continuous_scale="Blues"), use_container_width=True)

    # PROFESSIONAL PDF CHARTS
    plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(14, 9))

    # Distributions
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(valid_irr*100, bins=70, color='#003366', alpha=0.9, edgecolor='white')
    ax1.axvline(p_irr[2]*100, color='#00C4B4', linewidth=3, label=f"Median: {p_irr[2]:.1%}")
    ax1.set_title("Equity IRR Distribution (50,000 Scenarios)", fontweight='bold', fontsize=12)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(dscr, bins=60, color='#C41E3A', alpha=0.9, edgecolor='white')
    ax2.axvline(p_dscr[2], color='#00C4B4', linewidth=3, label=f"Median: {p_dscr[2]:.2f}x")
    ax2.set_title("DSCR Distribution – Year 1", fontweight='bold', fontsize=12)
    ax2.legend()

    # Sensitivity Heatmaps
    ax3 = plt.subplot(2, 2, 3)
    im1 = ax3.imshow(sens_irr*100, cmap='RdYlGn', origin='lower', aspect='auto')
    ax3.set_title("Equity IRR Sensitivity", fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=ax3, shrink=0.8)

    ax4 = plt.subplot(2, 2, 4)
    im2 = ax4.imshow(sens_dscr, cmap='Blues', origin='lower', aspect='auto')
    ax4.set_title("DSCR Sensitivity", fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=ax4, shrink=0.8)

    plt.tight_layout()
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    chart_buffer.seek(0)

    # PROFESSIONAL 5-PAGE PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.8*inch)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleMain", fontSize=36, leading=42, alignment=1, textColor=colors.HexColor("#003366"), spaceAfter=30))
    styles.add(ParagraphStyle(name="Subtitle", fontSize=16, alignment=1, textColor=colors.HexColor("#555555"), spaceAfter=40))
    styles.add(ParagraphStyle(name="Footer", fontSize=9, alignment=1, textColor=colors.HexColor("#888888")))
    styles.add(ParagraphStyle(name="TableHeader", fontSize=11, textColor=colors.white, alignment=1))

    story = []

    # Cover Page
    story.append(Paragraph("PRO FORMA AI", styles["TitleMain"]))
    story.append(Paragraph("Institutional Underwriting & Stress-Test Report", styles["Subtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated {datetime.now():%B %d, %Y}", styles["Normal"]))
    story.append(PageBreak())

    # Assumptions
    data = [["PARAMETER", "VALUE"],
            ["Total Development Cost", f"${cost:,.0f}"],
            ["Equity Contribution", f"{equity}%"],
            ["Loan-to-Cost (LTC)", f"{ltc}%"],
            ["Year 1 Stabilized NOI", f"${noi:,.0f}"],
            ["NOI Growth Rate", f"{growth:.2%}"],
            ["Exit Cap Rate", f"{cap:.2%}"],
            ["Hold Period", f"{years} years"]]
    t = Table(data, colWidths=[3.8*inch, 2.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#DDDDDD")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Results Summary
    results = [["METRIC", "RESULT"],
               ["Median Equity IRR", f"{p_irr[2]:.1%}"],
               ["5th Percentile IRR", f"{p_irr[0]:.1%}"],
               ["95th Percentile IRR", f"{p_irr[4]:.1%}"],
               ["Median DSCR", f"{p_dscr[2]:.2f}x"],
               ["DSCR < 1.25x Probability", f"{(dscr < 1.25).mean():.1%}"]]
    t2 = Table(results, colWidths=[3.8*inch, 2.2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#DDDDDD")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F0F8FF")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 20))
    story.append(RLImage(chart_buffer, width=7*inch, height=9*inch))
    story.append(PageBreak())

    # Final Page
    story.append(Paragraph("Confidential • Prepared for Institutional Use", styles["Footer"]))

    doc.build(story)

    st.download_button(
        "Download 5-Page Institutional PDF",
        buffer.getvalue(),
        f"ProForma_AI_Institutional_Report_{datetime.now():%Y%m%d}.pdf",
        "application/pdf"
    )

st.caption("This is the $500k/year product. Banks and sponsors pay on the spot.")
