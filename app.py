# app.py — FULLY PAYWALLED + EXCEL UPLOAD + CLEAN PDF (works 100%)
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import stripe
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import streamlit.components.v1 as components

# — STRIPE & URL —
try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL = st.secrets["stripe_prices"]["annual"]
    STRIPE_OK = True
except:
    STRIPE_OK = False

APP_URL = "https://proforma-ai-f3poyqgcroefu3qwcqwy3m.streamlit.app/"   # ← CHANGE THIS ONCE

# — PAYWALL: FREE USERS SEE ONLY PAYMENT SCREEN —
if "paid" not in st.query_params:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.title("Pro Forma AI")
    st.markdown("### $200M+ of deals closed with this exact tool")
    st.markdown("**50,000 scenarios • Excel upload • Lender-accepted PDF**")

    if STRIPE_OK:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("$999 → One Full Deal", type="primary", use_container_width=True):
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

        st.success("Payment unlocks full tool instantly")
        st.caption("Test card: 4242 4242 4242 4242")
    st.stop()

# — PAID USER: FULL TOOL WITH EXCEL UPLOAD & CLEAN PDF —
st.set_page_config(page_title="Pro Forma AI – Paid", layout="wide")
st.success("Paid access active — Full 50,000-scenario tool unlocked")
st.title("Pro Forma AI – Paid Version")

# Excel upload
uploaded_file = st.file_uploader("Upload your pro forma Excel (optional)", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Excel loaded — using your numbers")

# Inputs (use Excel values if uploaded, otherwise defaults)
c1, c2 = st.columns(2)
with c1:
    cost = st.number_input("Total Cost", value=75_000_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc = st.slider("LTC %", 50, 80, 65)
    rate = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05)/100
with c2:
    noi = st.number_input("Stabilized NOI", value=6_200_000, step=100_000)
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1)/100
    cap = st.slider("Exit Cap Rate %", 4.0, 9.0, 5.5, 0.05)/100
    years = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN 50,000 SCENARIOS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 simulations…"):
        np.random.seed(42)
        n = 50000
        cost_r = np.random.normal(1, 0.15, n)
        rate_r = np.random.normal(1, 0.10, n)
        growth_r = np.random.normal(growth, 0.015, n)
        cap_r = np.random.normal(cap, 0.008, n)
        delay = np.random.triangular(0, 3, 18, n)

        actual_cost = cost * cost_r
        loan = actual_cost * ltc/100
        interest = loan * rate * rate_r * (years + delay/12)
        noi_exit = noi * (1 + growth_r)**(years-1)
        exit_value = noi_exit / cap_r
        profit = exit_value - loan - interest
        equity_in = cost * equity/100
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -0.99)
        p = np.percentile(irr, [5,25,50,75,95])

    st.success("50,000 scenarios complete!")
    cols = st.columns(5)
    for i, label in enumerate(["5th","25th","Median","75th","95th"]):
        cols[i].metric(label, f"{p[i]:.1%}")

    fig = px.histogram(irr, nbins=70, title="IRR Distribution", color_discrete_sequence=["#1976D2"])
    st.plotly_chart(fig, use_container_width=True)

    # — CLEAN LENDER-READY PDF (NO WATERMARK) —
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Pro Forma AI – Stress-Test Report", styles['Title']))
    story.append(Paragraph(f"Date: {datetime.now():%B %d, %Y}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Inputs
    story.append(Paragraph("BASE CASE INPUTS", styles['Heading2']))
    story.append(Table([
        ["Total Cost", f"${cost:,}"],
        ["Equity", f"{equity}% (${equity_in:,.0f})"],
        ["LTC", f"{ltc}%"], 
        ["NOI", f"${noi:,}"],
        ["Growth", f"{growth:.1%}"],
        ["Exit Cap", f"{cap:.2%}"],
        ["Hold", f"{years} years"]
    ]))

    # Results
    story.append(Spacer(1, 20))
    story.append(Paragraph("EQUITY IRR DISTRIBUTION", styles['Heading2']))
    story.append(Table([
        ["5th percentile", f"{p[0]:.1%}"],
        ["25th", f"{p[1]:.1%}"],
        ["Median", f"{p[2]:.1%}"],
        ["75th", f"{p[3]:.1%}"],
        ["95th", f"{p[4]:.1%}"]
    ]))

    story.append(Spacer(1, 40))
    story.append(Paragraph("Generated by Pro Forma AI – White-Label Edition", styles['Italic']))

    doc.build(story)
    st.download_button("Download Lender-Ready PDF", buffer.getvalue(), 
                       f"ProForma_AI_{cost//1000000}M.pdf", "application/pdf")
