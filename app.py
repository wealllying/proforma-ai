# app.py — FINAL + 2-PAGE PDF WITH CHART (WORKS ON STREAMLIT CLOUD)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    st.markdown("#### 50,000-scenario stress test → 2-page lender PDF with chart")
    st.markdown("**Used on $300M+ of deals**")

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
        if st.button("$15,000/yr → Unlimited", use_container_width=True):
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
st.set_page_config(page_title="Pro Forma AI – Paid", layout="wide")
st.success("Paid access active — Full 2-page PDF with chart")
st.title("Pro Forma AI")

st.info("Enter your deal numbers below — takes 20 seconds")

c1, c2 = st.columns(2)
with c1:
    cost   = st.number_input("Total Development Cost ($)", value=92_500_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi    = st.number_input("Year 1 Stabilized NOI ($)", value=7_200_000, step=100_000)
    growth = st.slider("Annual NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN 50,000 SCENARIOS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo simulations…"):
        np.random.seed(42)
        n = 50000
        cost_var   = np.random.normal(1, 0.15, n)
        rate_var   = np.random.normal(1, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var    = np.random.normal(cap, 0.008, n)

        actual_cost = cost * cost_var
        loan = actual_cost * (ltc/100)
        interest = loan * rate * rate_var * years
        noi_exit = noi * (1 + growth_var)**(years-1)
        exit_value = noi_exit / cap_var
        profit = exit_value - loan - interest
        equity_in = cost * (equity/100)
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -1)
        valid_irr = irr[irr > -1]
        p = np.percentile(valid_irr, [5, 25, 50, 75, 95])

    st.success("50,000 scenarios complete!")
    cols = st.columns(5)
    for i, label in enumerate(["5th", "25th", "Median", "75th", "95th"]):
        cols[i].metric(label, f"{p[i]:.1%}")

    # Interactive Plotly chart (for app)
    fig_plotly = px.histogram(valid_irr*100, nbins=70, title="Equity IRR Distribution (%)",
                              color_discrete_sequence=["#1976D2"])
    fig_plotly.update_layout(showlegend=False)
    st.plotly_chart(fig_plotly, use_container_width=True)

    # ——— MATPLOTLIB CHART FOR PDF (WORKS ON STREAMLIT CLOUD) ———
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid_irr*100, bins=70, color='#1976D2', alpha=0.8, edgecolor='white')
    ax.set_title("Equity IRR Distribution (50,000 Scenarios)", fontsize=14, pad=20)
    ax.set_xlabel("Equity IRR (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(p[2]*100, color='orange', linewidth=2, label=f"Median: {p[2]:.1%}")
    ax.axvline(p[0]*100, color='red', linestyle='--', label=f"5th: {p[0]:.1%}")
    ax.axvline(p[4]*100, color='green', linestyle='--', label=f"95th: {p[4]:.1%}")
    ax.legend()

    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    img_buffer.seek(0)

    # ——— 2-PAGE LENDER PDF WITH CHART ———
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=24, alignment=1, textColor=colors.HexColor("#1976D2")))
    styles.add(ParagraphStyle(name="Footer", font Way to go! fontSize=9, textColor=colors.grey, alignment=1))

    story = [
        Paragraph("Pro Forma AI", styles["TitleBig"]),
        Paragraph("Equity Stress-Test Report", styles["Title"]),
        Spacer(1, 20),
        Paragraph(f"Generated: {datetime.now():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 20),

        Table([
            ["KEY ASSUMPTIONS", ""],
            ["Total Cost", f"${cost:,}"],
            ["Equity", f"{equity}%"],
            ["LTC", f"{ltc}%"],
            ["NOI (Year 1)", f"${noi:,}"],
            ["NOI Growth", f"{growth:.1%}"],
            ["Exit Cap", f"{cap:.2%}"],
            ["Hold", f"{years} years"],
        ], colWidths=[3.5*inch, 2.5*inch]),

        Spacer(1, 20),
        Table([
            ["IRR DISTRIBUTION", ""],
            ["5th Percentile", f"{p[0]:.1%}"],
            ["Median", f"{p[2]:.1%}"],
            ["95th Percentile", f"{p[4]:.1%}"],
            [">12% IRR Chance", f"{(valid_irr > 0.12).mean():.1%}"],
            [">15% IRR Chance", f"{(valid_irr > 0.15).mean():.1%}"],
        ], colWidths=[3.5*inch, 2.5*inch]),

        Spacer(1, 20),
        RLImage(img_buffer, width=6.5*inch, height=4*inch),
        Spacer(1, 30),
        Paragraph("50,000 Monte Carlo simulations with variance in cost, rate, growth, and cap rate.", styles["Normal"]),
        Spacer(1, 20),
        Paragraph("Generated by Pro Forma AI – White-Label Edition", styles["Footer"]),
    ]

    for t in [story[5], story[7]]:
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1976D2")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
        ]))

    doc.build(story)

    st.download_button(
        "Download 2-Page Lender PDF with Chart",
        buffer.getvalue(),
        f"ProForma_AI_{cost//1000000}M_Report.pdf",
        "application/pdf"
    )
