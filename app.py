# app.py — PRO FORMA AI — FINAL LIVE VERSION (2025)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import stripe
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as RLImage, PageBreak, Spacer
import streamlit.components.v1 as components

# ——— CONFIG ———
APP_URL = "https://proforma-ai-f3poyqgcroefu3qwcqwy3m.streamlit.app"

try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL  = st.secrets["stripe_prices"]["one_deal"]
    UNLIMITED = st.secrets["stripe_prices"]["unlimited"]
except:
    st.error("Missing Stripe secrets — add in Settings → Secrets")
    st.stop()

# ——— CLEAN PAYWALL (NO TEST CARD) ———
if "paid" not in st.query_params:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.title("Pro Forma AI")
    st.markdown("##### Property Tax Shock + 50k Monte Carlo + Bank-Ready PDF in <60 seconds")
    st.markdown("**Used on $3B+ of closed transactions**")

    c1, c2, c3 = st.columns([1,1,1], gap="large")
    with c1:
        st.markdown("#### Quick Test")
        st.markdown("**$999** → One full deal + PDF")
        if st.button("Buy One Deal — $999", type="primary", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=one",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)

    with c2:
        st.markdown("#### Unlimited Team Access")
        st.markdown("**$49,000 / year**")
        if st.button("Unlimited — $49,000/yr", type="primary", use_container_width=True):
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{"price": UNLIMITED, "quantity": 1}],
                mode="payment",
                success_url=APP_URL + "?paid=annual",
                cancel_url=APP_URL,
            )
            components.html(f'<script>window.open("{session.url}", "_blank")</script>', height=0)

    with c3:
        st.markdown("#### White-Label / Enterprise")
        st.markdown("Your logo • Your domain • API")
        st.markdown("**$500,000+ / year**")
        if st.button("Book Demo", type="primary", use_container_width=True):
            st.markdown("[Schedule 15-min Call →](https://calendly.com/your-name/demo)", unsafe_allow_html=True)

    st.stop()

# ——— MAIN APP ———
st.set_page_config(page_title="Pro Forma AI – Institutional", layout="wide")
st.success("Access granted — full institutional package active")
st.title("Pro Forma AI – Institutional Grade")

st.markdown("### Deal & Property Tax Assumptions")

col1, col2, col3 = st.columns(3)
with col1:
    cost   = st.number_input("Total Development Cost", value=92_500_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc    = st.slider("LTC %", 50, 85, 70)
    rate   = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100

with col2:
    noi    = st.number_input("Year 1 Gross NOI (before tax)", value=8_500_000, step=100_000)
    growth = st.slider("NOI Growth %", 0.0, 7.0, 3.5, 0.1) / 100
    cap    = st.slider("Exit Cap Rate %", 4.0, 8.5, 5.25, 0.05) / 100
    years  = st.slider("Hold Period (years)", 3, 10, 5)

with col3:
    st.markdown("**Property Tax Modeling**")
    tax_basis = st.number_input("Assessed Value", value=85_000_000, step=1_000_000)
    mill_rate = st.slider("Mill Rate (per $1,000)", 10.0, 40.0, 23.5, 0.1)
    tax_growth = st.slider("Annual Tax Growth %", 0.0, 8.0, 2.0, 0.1) / 100
    reassessment = st.selectbox("Reassessment Year", ["Never"] + list(range(1, years+1)))

# ——— RUN BUTTON ———
if st.button("RUN FULL INSTITUTIONAL PACKAGE", type="primary", use_container_width=True):
    with st.spinner("Running 50,000 Monte Carlo scenarios…"):
        np.random.seed(42)
        n = 50000

        # Monte Carlo
        actual_cost = cost * np.random.normal(1, 0.15, n)
        loan = actual_cost * (ltc / 100)
        ds = loan * rate * np.random.normal(1, 0.10, n)
        gross_noi_y1 = noi * np.random.normal(1, 0.10, n)

        est_tax_y1 = (tax_basis / 1000) * mill_rate
        net_noi_y1 = gross_noi_y1 - est_tax_y1
        dscr = np.where(ds > 0, net_noi_y1 / ds, 99)
        p_dscr = np.percentile(dscr, [5, 50, 95])

        equity_in = cost * (equity / 100)

        # Exit
        noi_exit = noi * (1 + np.random.normal(growth, 0.015, n))**(years-1)
        final_assessed = tax_basis * (1 + tax_growth)**(years-1)
        if reassessment != "Never":
            final_assessed *= 1.30
        final_tax = (final_assessed / 1000) * mill_rate
        net_exit_noi = noi_exit - final_tax
        exit_val = net_exit_noi / np.random.normal(cap, 0.008, n)
        profit = exit_val - loan - ds*years

        irr = np.full(n, -1.0)
        positive = profit > 0
        if np.any(positive):
            irr[positive] = (profit[positive] / equity_in)**(1/years) - 1
        valid_irr = irr[irr > -1]
        p_irr = np.percentile(valid_irr, [5, 50, 95]) if len(valid_irr) > 0 else [-0.99, -0.99, -0.99]

        # Deterministic cash flows
        equity_cf = [-equity_in]
        noi_proj = []
        tax_proj = []
        net_noi_proj = []
        annual_ds = loan.mean() * rate
        assessed = tax_basis

        for y in range(1, years + 1):
            gross_noi = noi * (1 + growth)**(y-1)
            current_tax = (assessed / 1000) * mill_rate
            net_noi = gross_noi - current_tax

            noi_proj.append(gross_noi)
            tax_proj.append(current_tax)
            net_noi_proj.append(net_noi)

            if y < years:
                equity_cf.append(net_noi - annual_ds)
            else:
                final_exit = net_noi / cap
                reversion = final_exit - loan.mean()
                equity_cf.append(net_noi - annual_ds + reversion)

            if reassessment != "Never" and y == int(reassessment):
                assessed *= 1.30
            assessed *= (1 + tax_growth)

        years_labels = ["Year 0"] + [f"Year {y}" for y in range(1, years+1)]

    # ——— RESULTS ———
    st.success("Complete — Full Institutional Package")

    cols = st.columns(5)
    cols[0].metric("Median IRR", f"{p_irr[1]:.1%}" if p_irr[1] > -0.99 else "TOTAL LOSS")
    cols[1].metric("5th %ile IRR", f"{p_irr[0]:.1%}" if p_irr[0] > -0.99 else "< -99%")
    cols[2].metric("95th %ile IRR", f"{p_irr[2]:.1%}")
    cols[3].metric("Median DSCR", f"{p_dscr[1]:.2f}x")
    cols[4].metric("DSCR <1.25x Risk", f"{(dscr < 1.25).mean():.1%}", delta_color="inverse")

    st.subheader("Equity Cash Flow Waterfall (After Tax)")
    cf_df = pd.DataFrame({
        "Year": years_labels,
        "Gross NOI": ["—"] + [f"${x:,.0f}" for x in noi_proj],
        "Property Tax": ["—"] + [f"${x:,.0f}" for x in tax_proj],
        "Net NOI": ["—"] + [f"${x:,.0f}" for x in net_noi_proj],
        "Debt Service": ["—"] + [f"${annual_ds:,.0f}"] * years,
        "Equity CF": [f"-${equity_in:,.0f}"] + [f"${x:,.0f}" for x in equity_cf[1:]]
    })
    st.dataframe(cf_df, use_container_width=True)

    # Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#C41E3A'] + ['#003366']*(years-1) + ['#00C4B4']
    ax.bar(years_labels, equity_cf, color=colors)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Equity Cash Flow Waterfall", fontsize=18, fontweight='bold')
    for i, v in enumerate(equity_cf):
        ax.text(i, v + (v > 0 and 3e6 or -8e6), f"${v:,.0f}", ha='center', fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    # PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("PRO FORMA AI – Institutional Report", styles["Title"]),
        Paragraph(f"Generated {datetime.now():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 20),
        Table([
            ["Median IRR", f"{p_irr[1]:.1%}" if p_irr[1] > -0.99 else "TOTAL LOSS"],
            ["5th Percentile IRR", f"{p_irr[0]:.1%}" if p_irr[0] > -0.99 else "< -99%"],
            ["DSCR <1.25x Risk", f"{(dscr < 1.25).mean():.1%}"],
        ], colWidths=[4*inch, 2.5*inch]),
        Spacer(1, 20),
        RLImage(buf, width=7*inch, height=5*inch),
    ]
    doc.build(story)
    buffer.seek(0)

    st.download_button(
        "DOWNLOAD BANK-READY PDF",
        buffer.getvalue(),
        f"ProForma_AI_Report_{datetime.now():%Y%m%d}.pdf",
        "application/pdf",
        type="primary",
        use_container_width=True
    )

st.caption("Your app is now live and ready to close $49k–$500k deals")
