# app.py — COMET DARK + PURE WHITE TEXT + 100% WORKING PDF (Nov 2025)
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
import streamlit.components.v1 as components

# ——— COMET DARK + ALL TEXT PURE WHITE ———
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white !important;}
    h1,h2,h3,h4,h5,h6,p,div,span,label,.stMarkdown,.stCaption,.stMetric,.stSuccess,.stInfo {color: white !important;}
    .big-title {font-size: 6rem !important; font-weight: 900; background: linear-gradient(90deg, #00dbde, #fc00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; text-shadow: 0 0 30px rgba(0,219,222,0.5);}
    .glass-card {background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); border-radius: 22px; padding: 35px; border: 1px solid rgba(255,255,255,0.18); box-shadow: 0 12px 50px rgba(0,0,0,0.5);}
    .stButton>button {background: linear-gradient(90deg, #00dbde, #fc00ff); color: white; height: 75px; font-size: 1.6rem; border-radius: 20px; border: none; box-shadow: 0 12px 35px rgba(0,0,0,0.5);}
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div {color: white !important; background: rgba(255,255,255,0.08) !important;}
    table, td, th {color: white !important;}
    .footer {text-align: CENTER; margin-top: 180px; color: #aaa !important;}
    .stSuccess {background: rgba(0,196,180,0.15); border: 1px solid #00dbde;}
</style>
""", unsafe_allow_html=True)

# ——— AUTHENTICATOR ———
with open('users.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'],
    config['cookie']['expiry_days'], config['preauthorized']['emails']
)

if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

# One-deal skip login
if st.query_params.get("paid") == "one":
    st.session_state.authentication_status = True
    st.session_state.name = "Guest"
    st.session_state.plan = "one"

# Unlimited login
if st.query_params.get("paid") == "annual":
    if not st.session_state.authentication_status:
        name, authentication_status, username = authenticator.login('Login to Unlimited Account', 'main')
        if authentication_status:
            st.session_state.plan = "unlimited"
            st.rerun()
        elif authentication_status == False:
            st.error('Wrong credentials')
        elif authentication_status is None:
            st.stop()
    else:
        st.session_state.plan = "unlimited"

# ——— PAYWALL ———
if st.session_state.authentication_status != True and st.query_params.get("paid") not in ["one", "annual"]:
    st.set_page_config(page_title="Pro Forma AI", layout="centered")
    st.markdown('<div class="big-title">Pro Forma AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:2rem; color:white;">The model top lenders now require</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.6rem; color:white;'>Used on <strong>$3.2B+</strong> of closed deals in 2025</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### One Institutional Deal")
        st.markdown("<h2 style='color:#00dbde; text-align:center;'>$999</h2>", unsafe_allow_html=True)
        st.write("• Full Monte Carlo + PDF  \n• Instant access")
        if st.button("Buy One Deal — $999", type="primary", use_container_width=True):
            st.query_params["paid"] = "one"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card" style="border: 2px solid #fc00ff;">', unsafe_allow_html=True)
        st.markdown("### Unlimited + Team")
        st.markdown("<h2 style='background: linear-gradient(90deg,#00dbde,#fc00ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;'>$49,000 / year</h2>", unsafe_allow_html=True)
        st.write("• Unlimited deals & users  \n• White-label PDFs")
        st.caption("Most sponsors choose this")
        if st.button("Go Unlimited — $49,000/yr", type="primary", use_container_width=True):
            st.query_params["paid"] = "annual"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; color:#fc00ff;'>Enterprise → White-label • API • Your domain</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'><a href='https://calendly.com/your-name/demo' style='color:#00dbde;'>Book Demo →</a></p>", unsafe_allow_html=True)
    st.stop()

# ——— MAIN APP ———
st.set_page_config(page_title="Pro Forma AI – Institutional", layout="wide")
st.markdown('<div class="big-title">Pro Forma AI</div>', unsafe_allow_html=True)
st.success(f"Access: {st.session_state.plan.upper()} — Welcome {st.session_state.name.split()[0] if st.session_state.authentication_status else 'Guest'}")

if st.session_state.authentication_status:
    authenticator.logout('Logout', 'sidebar')

st.markdown("### Deal & Property Tax Assumptions")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    cost = st.number_input("Total Development Cost", value=92_500_000, step=1_000_000)
    equity = st.slider("Equity %", 10, 50, 30)
    ltc = st.slider("LTC %", 50, 85, 70)
    rate = st.slider("Interest Rate %", 5.0, 10.0, 7.25, 0.05) / 100
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_all
