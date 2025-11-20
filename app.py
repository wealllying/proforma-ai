# app.py — FINAL MONEY-MAKING VERSION (Nov 2025)
import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import stripe

# — Safe Stripe —
try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL = st.secrets["stripe_prices"]["annual"]
    STRIPE_OK = True
except:
    STRIPE_OK = False

st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.title("Pro Forma AI — Real Estate Stress-Tester")
st.markdown("**50,000 Monte Carlo scenarios • Lender-ready PDF report**")

# — PAYMENT SIDEBAR —
with st.sidebar:
    st.header("Buy Instant Access")
    if STRIPE_OK:
        if st.button("$999 → One Full Deal", type="primary", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL, "quantity": 1}],
                mode="payment",
                success_url=f"{st.get_option('server.baseUrl')}/?paid=1",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location.href="{session.url}"</script>', unsafe_allow_html=True)

        if st.button("$15,000/yr → Unlimited", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ANNUAL, "quantity": 1}],
                mode="payment",
                success_url=f"{st.get_option('server.baseUrl')}/?paid=annual",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location.href="{session.url}"</script>', unsafe_allow_html=True)
        st.success("Payments LIVE")
    else:
        st.info("Free demo below")
        st.caption("Stripe secrets added → $999 & $15k buttons appear instantly")

# — INPUTS —
c1, c2 = st.columns(2)
with c1:
    cost = st.number_input("Total Development Cost", 30_000_000, 300_000_000, 75_000_000, 1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc = st.slider("Loan-to-Cost %", 50, 80, 65)
    rate = st.slider("Interest Rate", 5.0, 10.0, 7.25, 0.05) / 100
with c2:
    noi = st.number_input("Stabilized NOI", 2_000_000, 20_000_000, 6_200_000, 100_000)
    growth = st.slider("Annual Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap = st.slider("Exit Cap Rate", 3.5, 9.0, 5.5, 0.05) / 100
    years = st.slider("Hold Period (years)", 3, 10, 5)

if st.button("RUN 50,000 SCENARIOS", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo simulations…"):
        np.random.seed(42)
        n = 50000

        cost_risk   = np.random.normal(1, 0.15, n)
        rate_risk   = np.random.normal(1, 0.10, n)
        growth_risk = np.random.normal(growth, 0.015, n)
        cap_risk    = np.random.normal(cap, 0.008, n)
        delay_mo    = np.random.triangular(0, 3, 18, n)

        actual_cost = cost * cost_risk
        loan = actual_cost * ltc/100
        interest = loan * rate * rate_risk * (years + delay_mo/12)
        noi_exit = noi * (1 + growth_risk)**(years-1)
        exit_value = noi_exit / cap_risk
        profit = exit_value - loan - interest
        equity_in = cost * equity/100
        irr = np.where(profit > 0, (profit/equity_in)**(1/years) - 1, -0.99
