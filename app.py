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
# — FINAL, VERSION-PROOF STRIPE SIDEBAR —
with st.sidebar:
    st.header("Buy Instant Access")

    # $999 button
    if st.button("$999 → One Full Deal", type="primary", use_container_width=True):
        try:
            session = stripe.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": st.secrets["stripe_prices"]["one_deal"],
                    "quantity": 1
                }],
                mode="payment",
                success_url=st.get_option("server.baseUrl") + "/?paid=one",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<meta http-equiv="refresh" content="0; url={session.url}">', 
                     unsafe_allow_html=True)
        except Exception as e:
            st.error("Temporary glitch – copy error below for support")
            st.code(str(e))

    # $15,000 button
    if st.button("$15,000/year → Unlimited", use_container_width=True):
        try:
            session = stripe.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": st.secrets["stripe_prices"]["annual"],
                    "quantity": 1
                }],
                mode="payment",
                success_url=st.get_option("server.baseUrl") + "/?paid=annual",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<meta http-equiv="refresh" content="0; url={session.url}">', 
                     unsafe_allow_html=True)
        except Exception as e:
            st.error("Temporary glitch – copy error below for support")
            st.code(str(e))

    st.success("Payments LIVE")
    st.caption("Test card: 4242 4242 4242 4242 • any date • 123")
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

        # Random risk factors
        cost_risk   = np.random.normal(1.0, 0.15, n)
        rate_risk   = np.random.normal(1.0, 0.10, n)
        growth_risk = np.random.normal(growth, 0.015, n)
        cap_risk    = np.random.normal(cap, 0.008, n)
        delay_mo    = np.random.triangular(0, 3, 18, n)

        actual_cost = cost * cost_risk
        loan        = actual_cost * (ltc / 100)
        interest    = loan * rate * rate_risk * (years + delay_mo / 12)
        noi_exit    = noi * (1 + growth_risk) ** (years - 1)
        exit_value  = noi_exit / cap_risk
        profit      = exit_value - loan - interest
        equity_in   = cost * (equity / 100)

        irr = np.where(
            profit > 0,
            (profit / equity_in) ** (1 / years) - 1,
            -0.99
        )

        p = np.percentile(irr, [5, 25, 50, 75, 95])

    # ───── Results (outside the spinner) ─────
    st.success("50,000 scenarios complete!")
    cols = st.columns(5)
    for i, label in enumerate(["5th", "25th", "Median", "75th", "95th"]):
        cols[i].metric(label, f"{p[i]:.1%}")

    fig = px.histogram(irr, nbins=70, title="IRR Distribution (50,000 runs)",
                       color_discrete_sequence=["#1976D2"])
    st.plotly_chart(fig, use_container_width=True)

    # ───── Lender-ready report ─────
    report = f"""
    <html><body style="font-family:Arial;padding:40px;line-height:1.6">
    <h1 style="color:#1976D2">Pro Forma AI – Stress-Test Report</h1>
    <p><strong>{datetime.now():%B %d, %Y}</strong> • 50,000 scenarios • ${cost:,} deal</p>
    <table width="100%" cellpadding="12" style="border-collapse:collapse;font-size:18px">
    <tr style="background:#1976D2;color:white"><th>Percentile</th><th>IRR</th></tr>
    <tr><td>5th (severe)</td><td><strong>{p[0]:.1%}</strong></td></tr>
    <tr><td>25th</td><td>{p[1]:.1%}</td></tr>
    <tr style="background:#BBDEFB"><td>50th (median)</td><td><strong>{p[2]:.1%}</strong></td></tr>
    <tr><td>75th</td><td>{p[3]:.1%}</td></tr>
    <tr style="background:#E8F5E9"><td>95th (upside)</td><td><strong>{p[4]:.1%}</strong></td></tr>
    </table>
    <br><small>Generated instantly by Pro Forma AI</small>
    </body></html>
    """
    st.download_button(
        "Download Lender-Ready Report → Print → Save as PDF",
        report,
        f"ProForma_AI_Report_{cost//1000000}M.html",
        "text/html"
    )
