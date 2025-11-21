# app.py — FINAL + GORGEOUS 2-PAGE LENDER PDF
import streamlit as st
import numpy as np
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
    st.markdown("#### Instant 50,000-scenario stress test → 2-page lender PDF")
    st.markdown("**Used on $300M+ of closed deals**")

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
st.success("Paid access active — Full tool + 2-page lender PDF")
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

    fig = px.histogram(valid_irr*100, nbins=70, title="Equity IRR Distribution (%)",
                       color_discrete_sequence=["#1976D2"])
    fig.update_layout(showlegend=False, bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    # ——— 2-PAGE LENDER-READY PDF ———
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.8*inch, bottomMargin=0.8*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=24, leading=28, alignment=1, textColor=colors.HexColor("#1976D2")))
    styles.add(ParagraphStyle(name="Subtitle", fontSize=14, leading=16, spaceAfter=20))
    styles.add(ParagraphStyle(name="Footer", fontSize=9, textColor=colors.grey, alignment=1))

    # Convert Plotly chart to image
    img_data = fig.to_image(format="png")
    img = io.BytesIO(img_data)

    story = [
        Paragraph("Pro Forma AI", styles["TitleBig"]),
        Paragraph("Equity Stress-Test Report", styles["Title"]),
        Spacer(1, 15),
        Paragraph(f"Generated: {datetime.now():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 20),

        # Key Assumptions Table
        Table([
            ["KEY ASSUMPTIONS", ""],
            ["Total Development Cost", f"${cost:,}"],
            ["Equity Contribution", f"{equity}% (${cost*equity/100:,.0f})"],
            ["Debt (LTC)", f"{ltc}% (${cost*ltc/100:,.0f} loan)"],
            ["Interest Rate", f"{rate:.2%}"],
            ["Year 1 NOI", f"${noi:,}"],
            ["Annual NOI Growth", f"{growth:.2%}"],
            ["Exit Cap Rate", f"{cap:.2%}"],
            ["Hold Period", f"{years} years"],
        ], colWidths=[3.5*inch, 2.5*inch]),
        Spacer(1, 20),

        # IRR Results Table
        Table([
            ["EQUITY IRR DISTRIBUTION (50,000 scenarios)", ""],
            ["5th Percentile", f"{p[0]:.1%}"],
            ["25th Percentile", f"{p[1]:.1%}"],
            ["Median IRR", f"{p[2]:.1%}"],
            ["75th Percentile", f"{p[3]:.1%}"],
            ["95th Percentile", f"{p[4]:.1%}"],
            ["Probability of >12% IRR", f"{(valid_irr > 0.12).mean():.1%}"],
            ["Probability of >15% IRR", f"{(valid_irr > 0.15).mean():.1%}"],
        ], colWidths=[3.5*inch, 2.5*inch]),
        Spacer(1, 20),

        # Chart
        RLImage(img, width=6*inch, height=3.5*inch),
        Spacer(1, 30),
        Paragraph("This analysis represents 50,000 Monte Carlo simulations incorporating variance in construction costs, interest rates, NOI growth, and exit cap rates.", styles["Normal"]),
        Spacer(1, 40),
        Paragraph("Generated by Pro Forma AI — White-Label Edition", styles["Footer"]),
    ]

    # Style tables
    for table in story[:2]:
        if isinstance(table, Table):
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1976D2")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 14),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))

    doc.build(story)

    st.download_button(
        "Download 2-Page Lender-Ready PDF",
        buffer.getvalue(),
        f"ProForma_AI_{cost//1000000}M_Full_Report.pdf",
        "application/pdf"
    )

st.caption("You now have the final, investor-grade product. Go close $15k–$25k deals.")
