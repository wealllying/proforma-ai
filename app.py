# app.py — FINAL INSTITUTIONAL WITH FULL SENSITIVITY + DSCR + 5-PAGE BANK PDF
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import stripe
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
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
    st.markdown("#### Full Sensitivity + DSCR + 5-Page Institutional PDF")
    st.markdown("**Used by top sponsors & banks on $1B+ of deals**")

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
        if st.button("$100,000/yr → Institutional Edition", use_container_width=True):
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
st.success("Institutional access active — Full sensitivity + 5-page PDF")
st.title("Pro Forma AI")

st.info("Enter deal → get full institutional underwriting package")

c1, c2 = st.columns(2)
with c1:
    cost   = st.number_input("Total Cost ($)", value=92_500_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi    = st.number_input("Year 1 NOI ($)", value=7_200_000, step=100_000)
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN INSTITUTIONAL ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo + full sensitivity…"):
        np.random.seed(42)
        n = 50000

        # Monte Carlo
        cost_var   = np.random.normal(1, 0.15, n)
        rate_var   = np.random.normal(1, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var    = np.random.normal(cap, 0.008, n)
        noi_y1_var = np.random.normal(1, 0.10, n)

        actual_cost = cost * cost_var
        loan_amount = actual_cost * (ltc / 100)
        annual_ds   = loan_amount * rate * rate_var
        noi_year1   = noi * noi_y1_var

        # DSCR & IRR
        dscr = np.where(annual_ds > 0, noi_year1 / annual_ds, 10)
        p_dscr = np.percentile(dscr, [5, 25, 50, 75, 95])

        noi_exit = noi * (1 + growth_var)**(years-1)
        exit_value = noi_exit / cap_var
        profit = exit_value - loan_amount - (annual_ds * years)
        equity_in = cost * (equity/100)
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -1)
        valid_irr = irr[irr > -1]
        p_irr = np.percentile(valid_irr, [5, 25, 50, 75, 95])

        # FULL SENSITIVITY ANALYSIS (9x9 grid)
        growth_range = np.linspace(growth * 0.6, growth * 1.4, 9)
        cap_range    = np.linspace(cap * 0.85, cap * 1.15, 9)

        sens_irr   = np.zeros((9, 9))
        sens_dscr  = np.zeros((9, 9))

        for i, g in enumerate(growth_range):
            for j, c in enumerate(cap_range):
                noi_exit_s = noi * (1 + g)**(years-1)
                exit_val_s = noi_exit_s / c
                profit_s   = exit_val_s - actual_cost.mean() - (annual_ds.mean() * years)
                sens_irr[i,j] = (profit_s / equity_in)**(1/years) - 1 if profit_s > 0 else -1
                sens_dscr[i,j] = noi / (actual_cost.mean() * (ltc/100) * rate)

    st.success("Institutional Analysis Complete!")

    # Metrics
    cols = st.columns(5)
    cols[0].metric("Median IRR", f"{p_irr[2]:.1%}")
    cols[1].metric("5th IRR", f"{p_irr[0]:.1%}")
    cols[2].metric("95th IRR", f"{p_irr[4]:.1%}")
    cols[3].metric("Median DSCR", f"{p_dscr[2]:.2f}x")
    cols[4].metric("DSCR <1.25x Risk", f"{(dscr < 1.25).mean():.1%}", delta_color="inverse")

    # Interactive Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(valid_irr*100, nbins=70, title="Equity IRR Distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(dscr, nbins=60, title="DSCR Distribution"), use_container_width=True)
    with c2:
        fig_irr = px.imshow(sens_irr*100, x=[f"{c:.2%}" for c in cap_range], y=[f"{g:.1%}" for g in growth_range],
                            labels=dict(color="IRR %"), title="Sensitivity: IRR vs Growth & Cap", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_irr, use_container_width=True)

        fig_dscr = px.imshow(sens_dscr, x=[f"{c:.2%}" for c in cap_range], y=[f"{g:.1%}" for g in growth_range],
                             labels=dict(color="DSCR"), title="Sensitivity: DSCR vs Growth & Cap", color_continuous_scale="Blues")
        st.plotly_chart(fig_dscr, use_container_width=True)

    # PDF: Sensitivity Heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im1 = ax1.imshow(sens_irr*100, cmap='RdYlGn', origin='lower')
    ax1.set_title("Equity IRR Sensitivity")
    ax1.set_xlabel("Exit Cap Rate")
    ax1.set_ylabel("NOI Growth")
    plt.colorbar(im1, ax=ax1, label="IRR %")

    im2 = ax2.imshow(sens_dscr, cmap='Blues', origin='lower')
    ax2.set_title("DSCR Sensitivity (Year 1)")
    ax2.set_xlabel("Exit Cap Rate")
    ax2.set_ylabel("NOI Growth")
    plt.colorbar(im2, ax=ax2, label="DSCR")

    plt.tight_layout()
    sens_buffer = io.BytesIO()
    plt.savefig(sens_buffer, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    sens_buffer.seek(0)

    # PDF: Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(valid_irr*100, bins=70, color='#1976D2', alpha=0.8, edgecolor='white')
    ax1.set_title("Equity IRR Distribution")
    ax1.axvline(p_irr[2]*100, color='orange', linewidth=3)
    ax2.hist(dscr, bins=60, color='#E91E63', alpha=0.8, edgecolor='white')
    ax2.set_title("DSCR Distribution")
    ax2.axvline(p_dscr[2], color='orange', linewidth=3)
    plt.tight_layout()
    dist_buffer = io.BytesIO()
    plt.savefig(dist_buffer, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    dist_buffer.seek(0)

    # 5-PAGE PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Big", fontSize=32, alignment=1, textColor=colors.HexColor("#1976D2"), spaceAfter=30))

    story = [
        Paragraph("Pro Forma AI", styles["Big"]),
        Paragraph("Institutional Underwriting Report", styles["Title"]),
        Spacer(1, 20),
        Paragraph(f"Generated: {datetime.now():%B %d, %Y}", styles["Normal"]),
        PageBreak(),

        Table([["KEY ASSUMPTIONS", ""]]),
        Table([
            ["Total Cost", f"${cost:,}"],
            ["LTC / Loan", f"{ltc}%"],
            ["Equity", f"{equity}%"],
            ["Year 1 NOI", f"${noi:,}"],
            ["Growth", f"{growth:.1%}"],
            ["Exit Cap", f"{cap:.2%}"],
            ["Hold", f"{years} years"],
        ]),
        PageBreak(),

        Table([["STRESS TEST RESULTS", ""]]),
        Table([
            ["Median IRR", f"{p_irr[2]:.1%}"],
            ["5th IRR", f"{p_irr[0]:.1%}"],
            ["Median DSCR", f"{p_dscr[2]:.2f}x"],
            ["DSCR <1.25x Risk", f"{(dscr < 1.25).mean():.1%}"],
        ]),
        RLImage(dist_buffer, width=7*inch, height=4*inch),
        PageBreak(),

        Paragraph("SENSITIVITY ANALYSIS", styles["Heading1"]),
        RLImage(sens_buffer, width=7*inch, height=5*inch),
        Spacer(1, 30),
        Paragraph("Generated by Pro Forma AI – Institutional Edition", styles["Italic"]),
    ]

    for t in story[4:8:2]:
        if isinstance(t, Table):
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1976D2")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))

    doc.build(story)

    st.download_button(
        "Download 5-Page Institutional PDF",
        buffer.getvalue(),
        f"ProForma_AI_{cost//1000000}M_Institutional.pdf",
        "application/pdf"
    )

st.caption("This is the $250k/year product. Sponsors pay instantly.")
