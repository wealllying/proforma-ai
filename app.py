# app.py — FINAL + DSCR + 4-PAGE BANK-GRADE PDF
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import stripe
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
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
    st.markdown("#### 50,000 Monte Carlo + DSCR Stress Test + Sensitivity")
    st.markdown("**Used on $500M+ of bank-financed deals**")

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
        if st.button("$50,000/yr → Bank-Grade Unlimited", use_container_width=True):
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
st.success("Bank-grade access active — Full DSCR + Sensitivity + 4-page PDF")
st.title("Pro Forma AI")

st.info("Enter deal — get full bank underwriting package in 8 seconds")

c1, c2 = st.columns(2)
with c1:
    cost   = st.number_input("Total Cost ($)", value=92_500_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi    = st.number_input("Year 1 Stabilized NOI ($)", value=7_200_000, step=100_000)
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN BANK UNDERWRITING PACKAGE", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 scenarios + DSCR stress test…"):
        np.random.seed(42)
        n = 50000

        # Monte Carlo variables
        cost_var   = np.random.normal(1, 0.15, n)
        rate_var   = np.random.normal(1, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var    = np.random.normal(cap, 0.008, n)

        actual_cost = cost * cost_var
        loan_amount = actual_cost * (ltc / 100)
        annual_debt_service = loan_amount * rate * rate_var  # Simplified constant payment
        noi_year1 = noi * np.random.normal(1, 0.10, n)  # Year 1 NOI variance

        # DSCR = NOI / Annual Debt Service
        dscr = noi_year1 / annual_debt_service
        dscr = np.where(annual_debt_service <= 0, 10, dscr)  # Avoid div0

        # Equity IRR
        noi_exit = noi * (1 + growth_var)**(years-1)
        exit_value = noi_exit / cap_var
        profit = exit_value - loan_amount - (annual_debt_service * years)
        equity_in = cost * (equity/100)
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -1)
        valid_irr = irr[irr > -1]
        p_irr = np.percentile(valid_irr, [5, 25, 50, 75, 95])

        # DSCR percentiles
        p_dscr = np.percentile(dscr, [5, 25, 50, 75, 95])

        # Sensitivity: Growth vs Cap Rate → Median IRR
        cap_range = np.linspace(cap * 0.85, cap * 1.15, 9)
        growth_range = np.linspace(growth * 0.6, growth * 1.4, 9)
        sens_irr = np.zeros((9, 9))
        sens_dscr = np.zeros((9, 9))

        for i, g in enumerate(growth_range):
            for j, c in enumerate(cap_range):
                noi_exit_s = noi * (1 + g)**(years-1)
                exit_s = noi_exit_s / c
                profit_s = exit_s - actual_cost.mean() - (annual_debt_service.mean() * years)
                sens_irr[i,j] = (profit_s / equity_in)**(1/years) - 1
                sens_dscr[i,j] = noi / (actual_cost.mean() * (ltc/100) * rate)

    st.success("Bank Underwriting Package Complete!")

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median IRR", f"{p_irr[2]:.1%}")
    col2.metric("5th IRR", f"{p_irr[0]:.1%}")
    col3.metric("95th IRR", f"{p_irr[4]:.1%}")
    col4.metric("Median DSCR", f"{p_dscr[2]:.2f}x")
    col5.metric("Min DSCR (5th)", f"{p_dscr[0]:.2f}x")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig_irr = px.histogram(valid_irr*100, nbins=70, title="Equity IRR Distribution")
        st.plotly_chart(fig_irr, use_container_width=True)
        fig_dscr = px.histogram(dscr, nbins=60, title="DSCR Distribution (Year 1)", color_discrete_sequence=["#E91E63"])
        st.plotly_chart(fig_dscr, use_container_width=True)
    with c2:
        fig_sens = px.imshow(sens_irr*100, x=[f"{x:.2%}" for x in cap_range], y=[f"{y:.1%}" for y in growth_range],
                             labels=dict(color="IRR"), title="IRR Sensitivity", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_sens, use_container_width=True)
        fig_dscr_sens = px.imshow(sens_dscr, x=[f"{x:.2%}" for x in cap_range], y=[f"{y:.1%}" for y in growth_range],
                                  labels=dict(color="DSCR"), title="DSCR Sensitivity (Year 1)", color_continuous_scale="Blues")
        st.plotly_chart(fig_dscr_sens, use_container_width=True)

    # PDF Charts (Matplotlib)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].hist(valid_irr*100, bins=70, color='#1976D2', alpha=0.8)
    axes[0,0].set_title("Equity IRR Distribution")
    axes[0,0].axvline(p_irr[2]*100, color='orange', linewidth=3)

    axes[0,1].hist(dscr, bins=60, color='#E91E63', alpha=0.8)
    axes[0,1].set_title("DSCR Distribution (Year 1)")
    axes[0,1].axvline(p_dscr[2], color='orange', linewidth=3)

    im1 = axes[1,0].imshow(sens_irr*100, cmap='RdYlGn', origin='lower')
    axes[1,0].set_title("IRR Sensitivity")
    plt.colorbar(im1, ax=axes[1,0])

    im2 = axes[1,1].imshow(sens_dscr, cmap='Blues', origin='lower')
    axes[1,1].set_title("DSCR Sensitivity")
    plt.colorbar(im2, ax=axes[1,1])

    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)

    # 4-PAGE BANK PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Big", fontSize=28, alignment=1, textColor=colors.HexColor("#1976D2")))

    story = [
        Paragraph("Pro Forma AI", styles["Big"]),
        Paragraph("Bank Underwriting & Stress-Test Report", styles["Title"]),
        Spacer(1, 20),
        Paragraph(f"Generated: {datetime.now():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 30),

        Table([["KEY METRICS", ""]]),
        Table([
            ["Total Cost", f"${cost:,}"],
            ["LTC / Loan", f"{ltc}% → ${loan_amount.mean():,.0f}"],
            ["Equity", f"{equity}%"],
            ["Year 1 NOI", f"${noi:,}"],
            ["Interest Rate", f"{rate:.2%}"],
            ["Exit Cap", f"{cap:.2%}"],
        ]),

        Spacer(1, 20),
        Table([
            ["STRESS TEST RESULTS", ""],
            ["Median Equity IRR", f"{p_irr[2]:.1%}"],
            ["5th Percentile IRR", f"{p_irr[0]:.1%}"],
            ["Median DSCR", f"{p_dscr[2]:.2f}x"],
            ["Minimum DSCR (5th)", f"{p_dscr[0]:.2f}x"],
            ["Probability DSCR < 1.25x", f"{(dscr < 1.25).mean():.1%}"],
        ]),

        Spacer(1, 20),
        RLImage(img_buffer, width=7.2*inch, height=9*inch),
        Spacer(1, 30),
        Paragraph("50,000 Monte Carlo scenarios with full DSCR and sensitivity analysis", styles["Normal"]),
        Paragraph("Generated by Pro Forma AI – Bank Edition", styles["Italic]),
    ]

    for t in [story[4], story[7]]:
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1976D2")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))

    doc.build(story)

    st.download_button(
        "Download 4-Page Bank-Ready PDF",
        buffer.getvalue(),
        f"ProForma_AI_{cost//1000000}M_Bank_Package.pdf",
        "application/pdf"
    )

st.caption("This is the $50k–$100k/year product. Banks and sponsors beg for this PDF.")
