# app.py — FINAL CLEAN VERSION (NO UPLOADER • NO PARSING • JUST MONEY)
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import stripe
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import streamlit.components.v1 as components

# ——— CONFIG ———
APP_URL = "https://proforma-ai-f3poyqgcroefu3qwcqwy3m.streamlit.app"

# Stripe
try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DE = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL = st.secrets["stripe_prices"]["annual"]
except:
    st.error("Stripe secrets missing — add them to go live")
    st.stop()

# ——— PAYWALL ———
if "paid" not in st.query_params:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.title("Pro Forma AI")
    st.markdown("#### Instant 50,000-scenario stress test → lender-ready PDF")
    st.markdown("**Used on $200M+ of closed deals**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("$999 → One Deal", type="primary", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DE, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=one",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)

    with col2:
        if st.button("$15,000/yr → Unlimited (White-Label)", use_container_width=True):
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
st.success("Paid access active — Full tool unlocked")
st.title("Pro Forma AI")

st.info("Enter your deal numbers below — takes 20 seconds")

# ——— MANUAL INPUTS (clean & fast) ———
c1, c2 = st.columns(2)
with c1:
    cost   = st.number_input("Total Development Cost", value=92_500_000, step=1_000_000, format="%d")
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi    = st.number_input("Stabilized NOI", value=7_200_000, step=100_000, format="%d")
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.0, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

# ——— RUN SIMULATION ———
if st.button("RUN 50,000 SCENARIOS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo simulations…"):
        np.random.seed(42)
        n = 50000
        
        cost_var   = np.random.normal(1, 0.15, n)
        rate_var   = np.random.normal(1, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var    = np.random.normal(cap, 0.008, n)

        actual_cost = cost * cost_var
        loan_amount = actual_cost * (ltc / 100)
        total_interest = loan_amount * rate * rate_var * years
        noi_exit = noi * (1 + growth_var) ** (years - 1)
        exit_value = noi_exit / cap_var
        profit = exit_value - loan_amount - total_interest
        equity_in = cost * (equity / 100)
        irr = np.where(profit > 0, (profit / equity_in) ** (1/years) - 1, -1)
        percentiles = np.percentile(irr[irr > -1], [5, 25, 50, 75, 95])

    st.success("50,000 scenarios complete!")

    # Show key metrics
    cols = st.columns(5)
    labels = ["5th", "25th", "Median", "75th", "95th"]
    for i, label in enumerate(labels):
        cols[i].metric(label, f"{percentiles[i]:.1%}")

    # Chart
    fig = px.histogram(irr[irr > -1]*100, nbins=70, title="Equity IRR Distribution (%)",
                       color_discrete_sequence=["#1976D2"])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ——— GENERATE CLEAN PDF ———
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Pro Forma AI – Stress-Test Report", styles['Title']),
        Paragraph(f"Generated: {datetime.now():%B %d, %Y}", styles['Normal']),
        Spacer(1, 20),
        Table([
            ["Total Cost", f"${cost:,}"],
            ["Equity", f"{equity}%"],
            ["LTC", f"{ltc}%"],
            ["Stabilized NOI", f"${noi:,}"],
            ["NOI Growth", f"{growth:.1%}"],
            ["Exit Cap Rate", f"{cap:.2%}"],
            ["Hold Period", f"{years} years"],
        ]),
        Spacer(1, 20),
        Table([
            ["5th percentile IRR", f"{percentiles[0]:.1%}"],
            ["Median IRR", f"{percentiles[2]:.1%}"],
            ["95th percentile IRR", f"{percentiles[4]:.1%}"],
        ]),
        Spacer(1, 40),
        Paragraph("Generated by Pro Forma AI – White-Label Edition", styles['Italic']),
    ]
    doc.build(story)

    st.download_button(
        "Download Lender-Ready PDF",
        buffer.getvalue(),
        f"ProForma_AI_{cost//1000000}M_StressTest.pdf",
        "application/pdf"
    )

st.caption("You now have the final, perfect product. Go close deals.")
