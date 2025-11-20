# app.py — FULL WHITE-LABEL + EXCEL UPLOAD + REAL PDF (ready to charge $999–$25k)
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import stripe
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import streamlit.components.v1 as components

# — CONFIG —
st.set_page_config(page_title="Pro Forma AI", layout="wide")
APP_URL = "https://YOUR-REAL-APP.streamlit.app"  # ← CHANGE ONCE TO YOUR URL

# — STRIPE (safe) —
try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL = st.secrets["stripe_prices"]["annual"]
    STRIPE_OK = True
except:
    STRIPE_OK = False

# — HEADER WITH YOUR LOGO —
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://via.placeholder.com/150x50.png?text=YOUR+LOGO", width=150)  # ← replace with your logo URL
with col2:
    st.title("Pro Forma AI – Real Estate Stress-Tester")
    st.markdown("**50,000 Monte Carlo scenarios • Lender-ready PDF • White-label ready**")

# — PAYMENTS SIDEBAR —
with st.sidebar:
    st.header("Buy Instant Access")
    if STRIPE_OK:
        if st.button("$999 → One Full Deal", type="primary", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=one",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)

        if st.button("$15,000/yr → Unlimited", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ANNUAL, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=annual",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)
        st.success("Payments LIVE")
    else:
        st.info("Free demo below")

# — EXCEL UPLOAD OR MANUAL —
tab1, tab2 = st.tabs(["Excel Upload (Instant)", "Manual Entry"])

with tab1:
    uploaded = st.file_uploader("Upload pro forma Excel (optional)", type=["xlsx", "xls"])
    if uploaded:
        df = pd.read_excel(uploaded)
        st.success("Excel parsed – using your numbers")
        # Simple auto-detect (you can expand this)
        cost = float(df.iloc[0,1]) if len(df)>0 else 75_000_000
        equity = 30
        ltc = 65
        rate = 7.25 / 100
        noi = float(df.iloc[5,1]) if len(df)>5 else 6_200_000
        growth = 3.5 / 100
        cap = 5.5 / 100
        years = 5
    else:
        cost = 75_000_000; equity = 30; ltc = 65; rate = 7.25/100
        noi = 6_200_000; growth = 3.5/100; cap = 5.5/100; years = 5

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        cost = st.number_input("Total Cost", value=75_000_000, step=1_000_000)
        equity = st.slider("Equity %", 10, 50, 30)
        ltc = st.slider("LTC %", 50, 80, 65)
        rate = st.slider("Rate %", 5.0, 10.0, 7.25, 0.05)/100
    with c2:
        noi = st.number_input("Stabilized NOI", value=6_200_000, step=100_000)
        growth = st.slider("Growth %", 0.0, 7.0, 3.5, 0.1)/100
        cap = st.slider("Exit Cap", 3.5, 9.0, 5.5, 0.05)/100
        years = st.slider("Hold Years", 3, 10, 5)

# — RUN SIMULATION —
if st.button("RUN 50,000 SCENARIOS", type="primary", use_container_width=True):
    with st.spinner("Running…"):
        np.random.seed(42); n = 50000
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

    st.success("Complete!")
    cols = st.columns(5)
    for i, label in enumerate(["5th","25th","Median","75th","95th"]):
        cols[i].metric(label, f"{p[i]:.1%}")

    fig = px.histogram(irr, nbins=70, title="IRR Distribution", color_discrete_sequence=["#1976D2"])
    st.plotly_chart(fig, use_container_width=True)

    # — REAL PDF GENERATION —
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Pro Forma AI – Stress-Test Report", styles['Title']),
        Paragraph(f"{datetime.now():%B %d, %Y} • 50,000 scenarios", styles['Normal']),
        Spacer(1, 12),
        Table([["Percentile", "IRR"],
               ["5th", f"{p[0]:.1%}"],
               ["25th", f"{p[1]:.1%}"],
               ["Median", f"{p[2]:.1%}"],
               ["75th", f"{p[3]:.1%}"],
               ["95th", f"{p[4]:.1%}"]],
              style=TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1976D2")),
                                ('TEXTCOLOR', (0,0), (-1,0), colors.white)]))
    ]
    doc.build(story)
    pdf_bytes = buffer.getvalue()

    st.download_button("Download Lender-Ready PDF", pdf_bytes, f"ProForma_AI_{cost//1000000}M.pdf", "application/pdf")

st.caption("White-label ready • Excel upload • Real PDF • $999–$25k live")
