import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import stripe

try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL_PRICE = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL_PRICE = st.secrets["stripe_prices"]["annual"]
    STRIPE_READY = True
except Exception:
    stripe.api_key = None
    ONE_DEAL_PRICE = None
    ANNUAL_PRICE = None
    STRIPE_READY = False

# ───── Page config & title ─────
st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.title("Real Estate Pro Forma Stress-Tester")
st.markdown("**50,000 Monte Carlo scenarios in < 12 seconds • Lender-ready report**")

# ───── Sidebar with safe payment buttons ─────
with st.sidebar:
    st.header("Unlock Full Pro Version")

    if STRIPE_READY and ONE_DEAL_PRICE:
        if st.button("Pay $999 → One Full Deal", type="primary", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL_PRICE, "quantity": 1}],
                mode="payment",
                success_url=st.get_option("server.baseUrl") + "/?paid=true",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location.href="{session.url}"</script>', unsafe_allow_html=True)

        if st.button("$15,000/year → Unlimited", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ANNUAL_PRICE, "quantity": 1}],
                mode="payment",
                success_url=st.get_option("server.baseUrl") + "/?paid=annual",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location.href="{session.url}"</script>', unsafe_allow_html=True)
    else:
        st.warning("Payment buttons temporarily disabled — free runs below")
        st.caption("Add Stripe secrets → buttons appear instantly")

    st.caption("Test card: 4242 4242 4242 4242 • any date • 123")

# ───── Main input tab ─────
col1, col2 = st.columns(2)
with col1:
    total_cost = st.number_input("Total Development Cost", value=75_000_000, step=1_000_000)
    equity_pct = st.slider("Equity %", 15, 50, 30)
    ltv = st.slider("Loan-to-Cost %", 50, 80, 65)
    rate = st.slider("Interest Rate %", 5.0, 9.5, 7.25, step=0.1) / 100
with col2:
    noi = st.number_input("Stabilized NOI (Year 3+)", value=6_200_000, step=100_000)
    growth = st.slider("Rent/NOI Growth %", 1.0, 6.0, 3.5, step=0.1) / 100
    cap = st.slider("Exit Cap Rate %", 4.0, 8.0, 5.5, step=0.05) / 100
    years = st.number_input("Hold Period (years)", 3, 10, 5)

if st.button("RUN 50,000 SCENARIOS →", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo simulations…"):
        np.random.seed(42)
        n = 50000

        cost_var = np.random.normal(1.0, 0.15, n)
        rate_var = np.random.normal(1.0, 0.10, n)
        growth_var = np.random.normal(growth, 0.015, n)
        cap_var = np.random.normal(cap, 0.008, n)
        delay_mo = np.random.triangular(0, 3, 18, n)

        actual_cost = total_cost * cost_var
        loan = actual_cost * (ltv/100)
        interest = loan * rate * rate_var * (years + delay_mo/12)
        noi_exit = noi * (1 + growth_var)**(years-1)
        exit_value = noi_exit / cap_var
        profit = exit_value - loan - interest
        equity_in = total_cost * (equity_pct/100)
        irr = np.where(profit > 0, (profit / equity_in)**(1/years) - 1, -0.99)

        p = np.percentile(irr, [5, 25, 50, 75, 95])
        p5, p25, p50, p75, p95 = [round(x, 4) for x in p]

    st.success("Done — 50,000 scenarios in seconds!")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("5th %", f"{p5:.1%}")
    c2.metric("25th %", f"{p25:.1%}")
    c3.metric("Median IRR", f"{p50:.1%}", delta="Target")
    c4.metric("75th %", f"{p75:.1%}")
    c5.metric("95th %", f"{p95:.1%}")

    fig = px.histogram(irr, nbins=80, title="IRR Distribution")
    st.plotly_chart(fig, use_container_width=True)

    report = f"""
    <html><body style="font-family:Arial;padding:40px;">
    <h1 style="color:#1976D2">Pro Forma AI – Stress-Test Report</h1>
    <p><strong>{datetime.now():%B %d, %Y}</strong> • 50,000 scenarios</p>
    <table width="100%" cellpadding="10" style="border-collapse:collapse;">
        <tr><td><strong>Total Cost</strong></td><td>${total_cost:,}</td></tr>
        <tr><td><strong>Equity</strong></td><td>{equity_pct}% (${equity_in:,.0f})</td></tr>
        <tr><td><strong>Stabilized NOI</strong></td><td>${noi:,}</td></tr>
    </table><br>
    <table width="100%" cellpadding="12" style="border-collapse:collapse;font-size:18px;">
        <tr style="background:#1976D2;color:white"><th>Percentile</th><th>IRR</th></tr>
        <tr style="background:#FFEBEE"><td>5th</td><td><strong>{p5:.1%}</strong></td></tr>
        <tr><td>25th</td><td>{p25:.1%}</td></tr>
        <tr style="background:#E3F2FD"><td>50th</td><td><strong>{p50:.1%}</strong></td></tr>
        <tr><td>75th</td><td>{p75:.1%}</td></tr>
        <tr style="background:#E8F5E9"><td>95th</td><td><strong>{p95:.1%}</strong></td></tr>
    </table>
    </body></html>
    """
    st.download_button("Download Lender-Ready Report → Print as PDF", report,
                       f"ProForma_AI_{total_cost//1000000}M.html", "text/html")
