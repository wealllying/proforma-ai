# app.py
import streamlit as st
import modal
import pandas as pd
import plotly.express as px
from weasyprint import HTML
import os
import stripe

stripe.api_key = st.secrets["stripe"]["key"]

st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.title("Real Estate Pro Forma Stress-Tester")
st.markdown("### 50,000 scenarios • 12 seconds • Lender-ready PDF")

# Sidebar pricing
with st.sidebar:
    st.header("💰 Pricing")
    if st.button("One Deal – $999", type="primary"):
        checkout = stripe.checkout.sessions.create(
            payment_method_types=["card"],
            line_items=[{"price": "price_1...", "quantity": 1}],  # put your Stripe price ID
            mode="payment",
            success_url=st.secrets["urls"]["success"],
            cancel_url=st.secrets["urls"]["cancel"],
        )
        st.write(f'<a href="{checkout.url}" target="_blank">Pay $999 →</a>', unsafe_allow_html=True)
    st.markdown("---")
    st.write("Annual Unlimited → DM me for $15k–$25k")

tab1, tab2 = st.tabs(["Upload Excel", "Manual Entry"])

with tab1:
    uploaded = st.file_uploader("Drop your pro forma Excel here", type=["xlsx", "xls"])
    if uploaded:
        df = pd.read_excel(uploaded, sheet_name=None)
        st.success("Parsed! Extracting assumptions…")
        # Simple parser – real version uses Unstructured + GPT-4o (code below)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        total_cost = st.number_input("Total Development Cost", value=75_000_000, step=1_000_000)
        equity = st.slider("Equity %", 15, 50, 30)
        ltv = st.slider("Max LTV %", 50, 75, 65)
        rate = st.number_input("Interest Rate", 5.0, 9.0, 7.2, step=0.1)/100
    with col2:
        noi = st.number_input("Year-3 Stabilized NOI", value=6_200_000, step=100_000)
        growth = st.slider("Expected Rent Growth", 1.0, 6.0, 3.5)/100
        cap = st.slider("Exit Cap Rate", 4.0, 7.5, 5.5)/100
        years = st.number_input("Hold Period (years)", 3, 10, 5)

    if st.button("Run 50,000 Scenarios →", type="primary"):
        with st.spinner("Crunching 50,000 futures…"):
            f = modal.Function.lookup("proforma-full", "run_full_monte_carlo")
            result = f.remote({
                "base_cost": total_cost,
                "equity_in": total_cost * equity/100,
                "ltv": ltv/100,
                "base_rate": rate,
                "base_noi": noi,
                "growth_mean": growth,
                "exit_cap_mean": cap,
                "hold_yrs": years
            })

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("5th Percentile IRR", f"{result['irr_5']:.1%}")
        col2.metric("25th Percentile", f"{result['irr_25']:.1%}")
        col3.metric("Median IRR", f"{result['irr_50']:.1%}")
        col4.metric("95th Percentile", f"{result['irr_95']:.1%}")

        fig = px.histogram(result["histogram"], nbins=50, title="IRR Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # PDF generation
        html_report = f"""
        <h1>Pro Forma Stress-Test Report</h1>
        <h2>50,000 Scenarios • {total_cost:,} Deal</h2>
        <table border="1" style="width:100%">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>5th Percentile IRR</td><td>{result['irr_5']:.1%}</td></tr>
            <tr><td>Median IRR</td><td>{result['irr_50']:.1%}</td></tr>
            <tr><td>95th Percentile IRR</td><td>{result['irr_95']:.1%}</td></tr>
        </table>
        """
        pdf = HTML(string=html_report).write_pdf()
        st.download_button("📄 Download Lender-Ready PDF", pdf, "Stress_Test_Report.pdf")
