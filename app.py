import streamlit as st
import numpy as np
import plotly.express as px
from datetime import datetime
import stripe

# Safe Stripe setup
try:
    stripe.api_key = st.secrets["stripe"]["secret_key"]
    ONE_DEAL = st.secrets["stripe_prices"]["one_deal"]
    ANNUAL = st.secrets["stripe_prices"]["annual"]
    STRIPE_OK = True
except:
    STRIPE_OK = False

st.set_page_config(page_title="Pro Forma AI", layout="wide")
st.title("Pro Forma AI – Real Estate Stress-Tester")
st.markdown("**50,000 Monte Carlo scenarios • Lender-ready report**")

# PAYMENT SIDEBAR
with st.sidebar:
    st.header("Buy Instant Access")
    if STRIPE_OK:
        if st.button("$999 → One Deal (Instant)", type="primary", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ONE_DEAL, "quantity": 1}],
                mode="payment",
                success_url=f"{st.get_option('server.baseUrl')}/?paid=1",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location="{session.url}"</script>', unsafe_allow_html=True)

        if st.button("$15,000/year → Unlimited", use_container_width=True):
            session = stripe.checkout.sessions.create(
                payment_method_types=["card"],
                line_items=[{"price": ANNUAL, "quantity": 1}],
                mode="payment",
                success_url=f"{st.get_option('server.baseUrl')}/?paid=annual",
                cancel_url=st.get_option("server.baseUrl"),
            )
            st.write(f'<script>window.top.location="{session.url}"</script>', unsafe_allow_html=True)
    else:
        st.info("Free demo below")
        st.caption("Add Stripe secrets → buttons activate instantly")

# Your full working calculator (same as before — just paste your final calculator code here)
# ... [all the inputs + Monte Carlo + report from the working version]

# (I'll paste the full calculator code in the next message if you want — just say "full code")
