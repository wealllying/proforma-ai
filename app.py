# app.py — Pro Forma AI — Institutional (Fixed Stripe + PDF chart export + Monte Carlo fig grouping)
# Run: streamlit run app.py
import os
import io
import math
import textwrap
import base64
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import requests

# Optional PDF libs
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Try to ensure Plotly can export images (kaleido)
KALEIDO_AVAILABLE = True
try:
    import kaleido  # noqa: F401
except Exception:
    KALEIDO_AVAILABLE = False

# Safe numpy_financial fallback (works in Streamlit Cloud)
try:
    import numpy_financial as npf
except Exception:
    import numpy as _np
    npf = _np.financial if hasattr(_np, "financial") else None
    if npf is None:
        # Minimal implementations used earlier
        def pmt(rate, nper, pv, fv=0, when='end'):
            if rate == 0:
                return -(pv + fv) / nper
            x = 1 + rate
            pow_x = x ** nper
            return -(fv + pv * pow_x) * rate / (pow_x - 1) / (1 + rate * (when == 'begin'))
        def irr(values):
            def npv(r):
                return sum(v / ((1 + r) ** i) for i, v in enumerate(values))
            r = 0.1
            for _ in range(120):
                f = npv(r)
                if abs(f) < 1e-8:
                    return r
                # numeric derivative (finite diff)
                df = (npv(r + 1e-6) - npv(r - 1e-6)) / 2e-6
                if df == 0:
                    break
                r -= f / df
            return r
        npf = type("npf", (), {"pmt": pmt, "irr": irr})()

# --------------------------
# Configuration (env vars)
# --------------------------
STRIPE_PK = os.environ.get("STRIPE_PK", "").strip()
APP_URL = os.environ.get("APP_URL", "https://proforma-ai-production.up.railway.app/").rstrip("/") + "/"

# Basic checks
if not STRIPE_PK:
    stripe_ready = False
else:
    stripe_ready = True

# Streamlit page config
st.set_page_config(page_title="Pro Forma AI — Institutional (Full)", layout="wide")

# --------------------------
# Small helpers
# --------------------------
def export_plotly_to_png_bytes(fig, width=1000, height=600, scale=1):
    """
    Convert a Plotly figure to PNG bytes. Uses fig.to_image (kaleido) when available.
    Returns bytes or None.
    """
    try:
        # fig.to_image returns bytes
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return img_bytes
    except Exception:
        try:
            # fallback via plotly.io
            img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            return img_bytes
        except Exception:
            return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# --------------------------
# Session state defaults
# --------------------------
if "pending_checkout" not in st.session_state:
    st.session_state.pending_checkout = None  # dict with keys price, success_url, cancel_url

# --------------------------
# Pricing & tokens (replace with your real IDs)
# --------------------------
ONE_DEAL_PRICE_ID = os.environ.get("ONE_DEAL_PRICE_ID", "price_1SVfkUH2h13vRbN8zuo69kgv")
ANNUAL_PRICE_ID = os.environ.get("ANNUAL_PRICE_ID", "price_1SXqY7H2h13vRbN8k0wC7IEx")

VALID_TOKENS = {
    "one": os.environ.get("ONE_TOKEN", "supersecret-onedeal-2025-x7k9p2m4v8q1r5t3"),
    "annual": os.environ.get("ANNUAL_TOKEN", "supersecret-annual-2025-h4j6k8m1p3q5r7t9"),
}

# --------------------------
# Query params unlock check
# --------------------------
qp = st.query_params
plan = qp.get("plan", None)
token = qp.get("token", None)
if plan is not None:
    plan = plan[0]
if token is not None:
    token = token[0]

# If user already unlocked by query params, skip paywall. (simple check)
unlocked = (plan in VALID_TOKENS and token == VALID_TOKENS.get(plan))

# --------------------------
# PAYWALL UI (if not unlocked)
# --------------------------
if not unlocked:
    st.title("Pro Forma AI — Institutional Access Required")
    st.markdown("### Unlock Full Model Instantly")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("One Deal — $999", use_container_width=True, key="one_button"):
            st.session_state.pending_checkout = {
                "price": ONE_DEAL_PRICE_ID,
                "success_url": f"{APP_URL}?plan=one&token={VALID_TOKENS['one']}",
                "cancel_url": APP_URL
            }

    with col2:
        if st.button("Unlimited — $99,000/year", use_container_width=True, key="annual_button"):
            st.session_state.pending_checkout = {
                "price": ANNUAL_PRICE_ID,
                "success_url": f"{APP_URL}?plan=annual&token={VALID_TOKENS['annual']}",
                "cancel_url": APP_URL
            }

    # Also provide a fallback custom checkout button that calls createCheckoutSession on your backend
    st.markdown("---")
    st.info("We redirect to Stripe Checkout (client-side). Make sure `STRIPE_PK` is set in environment variables.")

    # If pending_checkout set, inject Stripe JS redirect (client-side)
    if st.session_state.pending_checkout:
        if not stripe_ready:
            st.error("Stripe not configured. Set STRIPE_PK as an environment variable (publishable key).")
        else:
            checkout = st.session_state.pending_checkout
            # Build JS that uses Stripe JS to redirect to Checkout
            js = f"""
            <script src="https://js.stripe.com/v3/"></script>
            <script>
            (function() {{
                var stripe = Stripe("{STRIPE_PK}");
                stripe.redirectToCheckout({{
                    lineItems: [{{ price: "{checkout['price']}", quantity: 1 }}],
                    mode: 'payment',
                    successUrl: "{checkout['success_url']}",
                    cancelUrl: "{checkout['cancel_url']}"
                }}).then(function(result) {{
                    if (result.error) {{
                        // display a small error element
                        var el = document.createElement('div');
                        el.style.padding = '12px';
                        el.style.background = '#fee';
                        el.style.border = '1px solid #f99';
                        el.style.position = 'fixed';
                        el.style.left = '12px';
                        el.style.top = '12px';
                        el.innerText = result.error.message || 'Stripe checkout error';
                        document.body.appendChild(el);
                    }}
                }});
            }})();
            </script>
            """
            st.components.v1.html(js, height=300)
            # Clear pending so it doesn't re-run repeatedly
            st.session_state.pending_checkout = None

        # stop rest of app until user completes payment/unlocks
        st.stop()

# --------------------------
# Sidebar with inputs (your existing UI)
# --------------------------
with st.sidebar:
    st.header("Acquisition & Capital Stack")
    purchase_price = st.number_input("Purchase Price ($)", value=100_000_000, step=1_000_000)
    closing_costs_pct = st.slider("Closing Costs %", 0.0, 5.0, 1.5) / 100.0
    total_cost = purchase_price * (1 + closing_costs_pct)

    st.subheader("Senior Loan")
    senior_ltv = st.slider("Senior LTV %", 0.0, 90.0, 60.0) / 100.0
    senior_rate = st.slider("Senior Interest Rate (annual %)", 0.5, 12.0, 5.5, 0.05) / 100.0
    senior_amort = st.number_input("Senior Amortization (years, 0=IO)", 0, 30, 25)
    senior_term = st.number_input("Senior Term (years)", 1, 30, 25)
    senior_io = st.number_input("Senior IO Period (years)", 0, min(10, senior_term), 0)

    st.subheader("Mezz / Subordinate (optional)")
    use_mezz = st.checkbox("Include Mezz", value=False)
    if use_mezz:
        mezz_pct = st.slider("Mezz % of Cost", 0.0, 40.0, 10.0) / 100.0
        mezz_rate = st.slider("Mezz Rate %", 0.0, 20.0, 10.0) / 100.0
        mezz_term = st.number_input("Mezz Term (years)", 1, 10, 5)
    else:
        mezz_pct = 0.0
        mezz_rate = 0.0
        mezz_term = 0

    st.subheader("Equity")
    total_equity = total_cost * (1 - senior_ltv - (mezz_pct if use_mezz else 0.0))
    st.markdown(f"Estimated equity required: **${total_equity:,.0f}**")
    lp_share_default = st.slider("LP % of Equity (rest is GP)", 50, 95, 80) / 100.0

    st.header("Operating Assumptions")
    gpr_y1 = st.number_input("Year 1 GPR ($)", value=12_000_000, step=10000)
    rent_growth = st.slider("Rent Growth %", 0.0, 8.0, 3.0, 0.1) / 100.0
    vacancy = st.slider("Vacancy %", 0.0, 20.0, 5.0, 0.1) / 100.0
    opex_y1 = st.number_input("Year 1 OpEx ($)", value=3_600_000, step=10000)
    opex_growth = st.slider("OpEx Growth %", 0.0, 8.0, 2.5, 0.1) / 100.0
    reserves = st.number_input("Annual Reserves/CapEx ($)", value=400_000, step=10000)

    st.header("Exit & Waterfall")
    hold = st.slider("Hold Period (years)", 1, 10, 5)
    exit_cap = st.slider("Exit Cap %", 3.0, 12.0, 5.5, 0.05) / 100.0
    selling_costs = st.slider("Selling Costs %", 0.0, 8.0, 5.0) / 100.0
    pref_annual = st.slider("Preferred Return (LP annual %)", 0.0, 15.0, 8.0, 0.1) / 100.0
    catchup_pct = st.slider("Catch-up % to GP after pref", 0.0, 100.0, 0.0, 1.0) / 100.0

    st.markdown("### Promote tiers (IRR-hurdle driven)")
    use_promote = st.checkbox("Enable promote tiers", True)
    if use_promote:
        tier1_hurdle = st.number_input("Tier1 Hurdle IRR (%)", value=12.0, step=0.5) / 100.0
        tier1_gp = st.number_input("Tier1 GP % of residual", value=30.0, step=1.0) / 100.0
        tier2_hurdle = st.number_input("Tier2 Hurdle IRR (%)", value=20.0, step=0.5) / 100.0
        tier2_gp = st.number_input("Tier2 GP % of residual", value=50.0, step=1.0) / 100.0
        promote_tiers = [(tier1_hurdle, tier1_gp), (tier2_hurdle, tier2_gp)]
    else:
        promote_tiers = None

    st.header("Monte Carlo & Performance")
    n_sims = st.number_input("Monte Carlo sims", min_value=500, max_value=20000, value=5000, step=500)
    sigma_rent = st.slider("Rent vol (σ)", 0.0, 0.25, 0.02, 0.005)
    sigma_opex = st.slider("OpEx vol (σ)", 0.0, 0.25, 0.015, 0.005)
    sigma_cap = st.slider("Cap vol (σ)", 0.0, 0.10, 0.004, 0.001)
    corr = np.array([[1.0, 0.2, -0.4],
                     [0.2, 1.0, -0.2],
                     [-0.4, -0.2, 1.0]])

    st.header("Report branding")
    logo_mode = st.selectbox("Logo input", options=["None", "Upload file (PNG/JPG)", "Provide image URL"])
    logo_file = None
    logo_url = None
    if logo_mode == "Upload file (PNG/JPG)":
        logo_file = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg"])
    elif logo_mode == "Provide image URL":
        logo_url = st.text_input("Logo image URL (https://...)")

# --------------------------
# Model helper functions (kept & updated)
# --------------------------
def robust_irr(cfs):
    try:
        irr_val = npf.irr(cfs)
        if irr_val is None or np.isnan(irr_val) or np.isinf(irr_val):
            raise Exception("npf failed")
        return float(irr_val)
    except Exception:
        def npv(r):
            return sum(cf / ((1 + r) ** i) for i, cf in enumerate(cfs))
        low, high = -0.9999, 10.0
        f_low, f_high = npv(low), npv(high)
        it = 0
        while f_low * f_high > 0 and it < 60:
            high *= 2
            f_high = npv(high)
            it += 1
        if f_low * f_high > 0:
            return float('nan')
        for _ in range(300):
            mid = (low + high) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-8:
                return mid
            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2

def annual_payment(loan, rate, amort_years):
    if amort_years == 0:
        return loan * rate
    return float(-npf.pmt(rate, amort_years, loan))

def safe_cap(rate):
    return min(max(rate, 0.03), 0.30)

def compute_amort_schedule(loan, rate, amort_years, years):
    balances, interests, principals, payments = [], [], [], []
    bal = loan
    if amort_years == 0:
        for y in range(1, years+1):
            interests.append(bal * rate)
            principals.append(0.0)
            payments.append(interests[-1])
            balances.append(bal)
        return balances, interests, principals, payments
    payment = annual_payment(loan, rate, amort_years)
    for y in range(1, years+1):
        interests.append(bal * rate)
        principal = min(max(payment - interests[-1], 0.0), bal)
        principals.append(principal)
        payments.append(interests[-1] + principal)
        balances.append(bal)
        bal = max(bal - principal, 0.0)
    return balances, interests, principals, payments

def settle_final_distribution(lp_cf_so_far, gp_cf_so_far, remaining_residual, equity_lp, promote_tiers):
    if remaining_residual <= 0 or promote_tiers is None or len(promote_tiers) == 0:
        lp_share = 0.8
        return remaining_residual * lp_share, remaining_residual * (1 - lp_share)
    lp_add_total = 0.0
    gp_add_total = 0.0
    residual_left = remaining_residual
    lp_so_far = [float(x) for x in lp_cf_so_far]
    for (hurdle, gp_pct) in promote_tiers:
        if residual_left <= 0:
            break
        irr_full = robust_irr(lp_so_far + [residual_left])
        if not np.isnan(irr_full) and irr_full >= hurdle:
            low, high = 0.0, residual_left
            for _ in range(80):
                mid = (low + high) / 2.0
                irr_mid = robust_irr(lp_so_far + [mid])
                if np.isnan(irr_mid):
                    low = mid
                    continue
                if irr_mid >= hurdle:
                    high = mid
                else:
                    low = mid
            X = high
            lp_add_total += X
            residual_left -= X
            gp_take = residual_left * gp_pct
            lp_take = residual_left - gp_take
            lp_add_total += lp_take
            gp_add_total += gp_take
            residual_left = 0.0
            break
        else:
            lp_add_total += residual_left
            residual_left = 0.0
            break
    if residual_left > 0:
        lp_add_total += residual_left * 0.8
        gp_add_total += residual_left * 0.2
    return lp_add_total, gp_add_total

def apply_periodic_waterfall(distributable, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct):
    lp_paid = 0.0
    gp_paid = 0.0
    rem = distributable
    if lp_roc_remaining > 0 and rem > 0:
        pay = min(lp_roc_remaining, rem)
        lp_paid += pay
        lp_roc_remaining -= pay
        rem -= pay
    if lp_pref_accrued > 0 and rem > 0:
        pay = min(lp_pref_accrued, rem)
        lp_paid += pay
        lp_pref_accrued -= pay
        rem -= pay
    if catchup_pct > 0 and rem > 0:
        gp_catch = rem * catchup_pct
        gp_paid += gp_catch
        rem -= gp_catch
    residual_left = rem
    return lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left

def build_model_and_settle_det():
    senior_loan = total_cost * senior_ltv
    mezz_loan = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan
    equity_lp = equity_total * lp_share_default
    equity_gp = equity_total - equity_lp
    years = []
    lp_cfs = [-equity_lp]
    gp_cfs = [-equity_gp]
    dscr_path = []
    bal = senior_loan
    lp_roc_remaining = equity_lp
    lp_pref_accrued = 0.0
    residual_accumulator = 0.0
    balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)
    for y in range(1, hold + 1):
        years.append(f"Year {y}")
        gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1))
        egi = gpr * (1 - vacancy)
        opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) + reserves
        noi = egi - opex
        tax = 0.0
        noi_at = noi - tax
        if senior_amort == 0 or y <= senior_io:
            interest = bal * senior_rate
            principal = 0.0
            payment = interest
        else:
            if y - 1 < len(payments):
                payment = payments[y - 1]
            else:
                payment = annual_payment(senior_loan, senior_rate, senior_amort)
            interest = bal * senior_rate
            principal = min(max(payment - interest, 0.0), bal)
        bal = max(bal - principal, 0.0)
        ds = interest + principal
        dscr = noi_at / ds if ds > 0 else 99.0
        dscr_path.append(dscr)
        op_cf = noi_at - ds
        lp_pref_accrued += equity_lp * pref_annual
        lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
            op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
        )
        lp_cfs.append(lp_paid)
        gp_cfs.append(gp_paid)
        residual_accumulator += residual_left
    exit_value = noi_at / safe_cap(exit_cap)
    exit_net = exit_value * (1 - selling_costs)
    exit_reversion = max(exit_net - bal, 0.0)
    final_residual = residual_accumulator + exit_reversion
    lp_add, gp_add = settle_final_distribution(lp_cfs, gp_cfs, final_residual, equity_lp, promote_tiers)
    lp_cfs[-1] += lp_add
    gp_cfs[-1] += gp_add
    cf_table = pd.DataFrame({
        "Period": ["Year 0"] + years,
        "LP CF": [lp_cfs[0]] + lp_cfs[1:],
        "GP CF": [gp_cfs[0]] + gp_cfs[1:],
    })
    return {
        "lp_cfs": lp_cfs,
        "gp_cfs": gp_cfs,
        "cf_table": cf_table,
        "dscr_path": dscr_path,
        "exit_value": exit_value,
        "exit_reversion": exit_reversion
    }

def run_montecarlo_return_fig(n_sims):
    """
    Run Monte Carlo, return (irrs_array, dscr_breach_count, fig_monte)
    fig_monte is a Plotly figure (not rendered inside the function).
    """
    cov = np.diag([sigma_rent ** 2, sigma_opex ** 2, sigma_cap ** 2])
    cov = np.sqrt(cov) @ corr @ np.sqrt(cov)
    try:
        L = np.linalg.cholesky(cov)
    except Exception:
        L = np.diag([sigma_rent, sigma_opex, sigma_cap])
    senior_loan = total_cost * senior_ltv
    mezz_loan_amt = total_cost * mezz_pct if (use_mezz and mezz_pct > 0) else 0.0
    equity_total = total_cost - senior_loan - mezz_loan_amt
    equity_lp = equity_total * lp_share_default
    equity_gp = equity_total - equity_lp
    lp_irrs = []
    dscr_breach_count = 0
    last_noi_at = 0.0
    for i in range(int(n_sims)):
        z = np.random.normal(size=3)
        shocks = L @ z
        rent_shock = 1.0 + shocks[0]
        opex_shock = 1.0 + shocks[1]
        cap_shock = shocks[2]
        bal = senior_loan
        lp_cf_sim = [-equity_lp]
        gp_cf_sim = [-equity_gp]
        lp_roc_remaining = equity_lp
        lp_pref_accrued = 0.0
        residual_acc = 0.0
        dscr_vals = []
        balances, interests, principals, payments = compute_amort_schedule(senior_loan, senior_rate, max(1, senior_amort), hold)
        for y in range(1, hold + 1):
            gpr = gpr_y1 * ((1 + rent_growth) ** (y - 1)) * rent_shock
            v = float(np.clip(vacancy + np.random.normal(0, 0.01), 0.0, 0.9))
            egi = gpr * (1 - v)
            opex = opex_y1 * ((1 + opex_growth) ** (y - 1)) * opex_shock + reserves
            noi = egi - opex
            tax = 0.0
            noi_at = noi - tax
            last_noi_at = noi_at
            if senior_amort == 0 or y <= senior_io:
                interest = bal * senior_rate
                principal = 0.0
                payment = interest
            else:
                if y - 1 < len(payments):
                    payment = payments[y - 1]
                else:
                    payment = annual_payment(senior_loan, senior_rate, senior_amort)
                interest = bal * senior_rate
                principal = min(max(payment - interest, 0.0), bal)
            bal = max(bal - principal, 0.0)
            ds = interest + principal
            dscr_val = noi_at / ds if ds > 0 else 99.0
            dscr_vals.append(dscr_val)
            op_cf = noi_at - ds
            lp_pref_accrued += equity_lp * pref_annual
            lp_paid, gp_paid, lp_roc_remaining, lp_pref_accrued, residual_left = apply_periodic_waterfall(
                op_cf, lp_roc_remaining, lp_pref_accrued, equity_lp, pref_annual, catchup_pct
            )
            lp_cf_sim.append(lp_paid)
            gp_cf_sim.append(gp_paid)
            residual_acc += residual_left
        cap_sim = safe_cap(exit_cap + cap_shock)
        exit_value = last_noi_at / cap_sim if cap_sim > 0 else 0.0
        exit_net = exit_value * (1 - selling_costs)
        exit_reversion = max(exit_net - bal, 0.0)
        final_residual = residual_acc + exit_reversion
        lp_add, gp_add = settle_final_distribution(lp_cf_sim, gp_cf_sim, final_residual, equity_lp, promote_tiers)
        lp_cf_sim[-1] += lp_add
        gp_cf_sim[-1] += gp_add
        irr_sim = robust_irr(lp_cf_sim)
        if not np.isnan(irr_sim) and irr_sim > -1:
            lp_irrs.append(irr_sim)
        if any(d < 1.2 for d in dscr_vals):
            dscr_breach_count += 1

    if len(lp_irrs) == 0:
        irrs = np.array([])
    else:
        irrs = np.array(lp_irrs)

    # Create histogram figure
    if irrs.size > 0:
        fig_monte = px.histogram(irrs * 100, nbins=80, title="LP IRR Distribution (Monte Carlo)", labels={'value': 'IRR %'})
        fig_monte.add_vline(x=np.percentile(irrs, 50) * 100, line_color="white", line_width=3)
    else:
        fig_monte = None

    return irrs, dscr_breach_count, fig_monte

# --------------------------
# PDF generation (embed exported PNGs)
# --------------------------
def add_header_footer(c, logo_blob_tuple):
    width, height = letter
    c.setFont("Helvetica", 8)
    c.drawString(36, height - 50, "Pro Forma AI — Institutional")
    c.drawRightString(width - 36, height - 50, datetime.today().strftime("%B %d, %Y"))
    page_num_text = f"Page {c.getPageNumber()}"
    c.drawCentredString(width / 2.0, 30, page_num_text)
    if logo_blob_tuple:
        try:
            logo_bytes, _ = logo_blob_tuple
            img = ImageReader(BytesIO(logo_bytes))
            c.drawImage(img, width - 140, height - 70, width=80, height=30, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

def generate_long_pdf_memo(det, monte_stats, png_images, logo_blob_tuple):
    """
    png_images: list of (bytes, title) where bytes are PNG bytes for charts to include
    """
    if not REPORTLAB_AVAILABLE:
        text = "ReportLab not available. Install reportlab to generate full PDF memo."
        return text.encode("utf-8"), f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.txt"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=72, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading1"]
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, leading=11)

    story.append(Paragraph("<b>Pro Forma AI — Institutional Memorandum</b>", heading))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {datetime.today().strftime('%B %d, %Y')}", normal))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_text = textwrap.fill(
        f"This memorandum summarizes a deterministic and Monte Carlo analysis for a proposed acquisition at a purchase price of ${purchase_price:,.0f}. "
        f"The analysis includes capital stack assumptions (senior LTV {senior_ltv:.0%}), mezzanine (if used), LP/GP split, preferred return, and a multi-tier promote based on IRR hurdles.",
        200)
    story.append(Paragraph(exec_text, normal))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    story.append(Paragraph("Investment Highlights", styles['Heading2']))
    bullets = [
        f"Purchase Price: ${purchase_price:,.0f}",
        f"Total Cost incl. closing: ${total_cost:,.0f}",
        f"Estimated Equity: ${total_equity:,.0f}",
        f"Hold Period: {hold} years",
        f"Exit Cap: {exit_cap:.2%}",
        f"Preferred Return (LP): {pref_annual:.2%}",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", normal))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    story.append(Paragraph("Key Assumptions", styles['Heading2']))
    asum_lines = [
        f"GPR Year 1: ${gpr_y1:,.0f}",
        f"Rent Growth (annual): {rent_growth:.2%}",
        f"Vacancy: {vacancy:.2%}",
        f"OpEx Year1: ${opex_y1:,.0f}",
        f"OpEx growth: {opex_growth:.2%}",
        f"Reserves/CapEx: ${reserves:,.0f}",
        f"Senior Rate: {senior_rate:.2%} | Senior Amort: {senior_amort} yrs | IO: {senior_io} yrs",
    ]
    for a in asum_lines:
        story.append(Paragraph(a, normal))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    story.append(Paragraph("Deterministic Cashflows (LP & GP)", styles['Heading2']))
    df = det['cf_table'].copy()
    df_formatted = df.copy()
    for col in ["LP CF", "GP CF"]:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.0f}")
    parts = [df_formatted.iloc[i:i+20] for i in range(0, len(df_formatted), 20)]
    for idx, part in enumerate(parts):
        data = [list(part.columns)] + part.values.tolist()
        t = Table(data, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        if idx < len(parts) - 1:
            story.append(PageBreak())

    # Add figures (png_images) after cashflows
    story.append(PageBreak())
    story.append(Paragraph("Figures", styles['Heading2']))
    story.append(Spacer(1, 8))
    for png_bytes, title in png_images:
        try:
            story.append(Paragraph(title, styles['Heading3']))
            story.append(Spacer(1, 6))
            img_reader = ImageReader(BytesIO(png_bytes))
            # The ImageReader inserted directly in story via a Paragraph is not straightforward,
            # so we use ReportLab's flowable technique: create a temporary table to hold the image
            im_table = Table([[img_reader]], colWidths=[450])
            im_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            # Instead of adding raw ImageReader, we add an Image to story using a canvas wrapper:
            # To keep it simple, use a paragraph telling user to view PDF if image fails.
            # Workaround: convert image to base64 and embed as raw HTML is not supported.
            # Simpler approach: use reportlab.platypus.Image via BytesIO
            from reportlab.platypus import Image as RLImage
            img_stream = BytesIO(png_bytes)
            rl_img = RLImage(img_stream, width=450, height=260)
            story.append(rl_img)
            story.append(Spacer(1, 12))
        except Exception:
            # if image adding fails gracefully continue
            story.append(Paragraph("Figure unavailable (image embedding failed).", normal))
            story.append(Spacer(1, 6))
    story.append(PageBreak())

    story.append(Paragraph("Monte Carlo Summary", styles['Heading2']))
    st_lines = [
        f"P5: {monte_stats.get('p5', 'N/A')}",
        f"P50: {monte_stats.get('p50', 'N/A')}",
        f"P95: {monte_stats.get('p95', 'N/A')}",
    ]
    for line in st_lines:
        story.append(Paragraph(line, normal))
    story.append(Spacer(1, 12))

    story.append(PageBreak())
    story.append(Paragraph("Appendix: Full Inputs & Notes", styles['Heading2']))
    inputs_summary = [
        f"Purchase price: ${purchase_price:,.0f}",
        f"Total Cost incl closing: ${total_cost:,.0f}",
        f"Senior LTV: {senior_ltv:.2%}",
        f"Mezz used: {use_mezz} (mezz %: {mezz_pct:.2%})",
        f"LP % of equity: {lp_share_default:.2%}",
        f"Hold: {hold} yrs",
    ]
    for i in inputs_summary:
        story.append(Paragraph(i, normal))
    story.append(Spacer(1, 12))
    story.append(Paragraph("CSV / Data Exports", styles['Heading3']))
    story.append(Paragraph("Download deterministic and Monte Carlo outputs from the web UI.", small))

    doc.build(story, onFirstPage=lambda c, d: add_header_footer(c, logo_blob_tuple),
              onLaterPages=lambda c, d: add_header_footer(c, logo_blob_tuple))
    buf.seek(0)
    return buf.getvalue(), f"Pro_Forma_AI_Memo_{datetime.today().strftime('%Y%m%d')}.pdf"

# --------------------------
# Main compute + UI flow
# --------------------------
st.title("Pro Forma AI — Institutional (Full)")

if st.button("Run Full Institutional Model (Deterministic + Monte Carlo)"):
    # Deterministic
    with st.spinner("Running deterministic build..."):
        det = build_model_and_settle_det()
        lp_irr = robust_irr(det["lp_cfs"])
    st.success("Deterministic build complete")

    # Prepare deterministic charts (Plotly)
    st.subheader("Deterministic Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("LP IRR (det)", f"{lp_irr:.2%}" if not math.isnan(lp_irr) else "N/A")
    c2.metric("Min DSCR", f"{min(det['dscr_path']):.2f}x" if det['dscr_path'] else "N/A")
    c3.metric("Exit Value (net)", f"${det['exit_value']*(1-selling_costs):,.0f}")

    st.markdown("### Deterministic Cashflow Table (LP)")
    st.dataframe(det["cf_table"])

    # Deterministic waterfall figure
    try:
        op_sum = sum([x for x in det['lp_cfs'][1:-1]]) if len(det['lp_cfs']) > 2 else 0
        wf = go.Figure(go.Waterfall(
            x=["Equity In", "Operating CF", "Exit/Residual"],
            y=[-abs(det['lp_cfs'][0]), op_sum, det['lp_cfs'][-1]],
            connector={"line": {"color": "white"}}
        ))
        wf.update_layout(title="Deterministic LP Waterfall", template="plotly_white", showlegend=False)
    except Exception:
        wf = None

    # Export deterministic charts to PNG list
    png_images = []
    if wf is not None:
        png = export_plotly_to_png_bytes(wf)
        if png:
            png_images.append((png, "Deterministic LP Waterfall"))

    # Monte Carlo
    st.info(f"Running Monte Carlo with {n_sims} sims — this may take time.")
    with st.spinner("Running Monte Carlo..."):
        irrs, breaches, fig_monte = run_montecarlo_return_fig(int(n_sims))

    if irrs.size == 0:
        st.error("Monte Carlo produced no valid IRRs. Check inputs.")
        p5 = p50 = p95 = None
    else:
        p5, p50, p95 = np.percentile(irrs, [5, 50, 95])
        st.subheader("Monte Carlo Results (LP IRR)")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("P5", f"{p5:.2%}")
        cc2.metric("P50", f"{p50:.2%}")
        cc3.metric("P95", f"{p95:.2%}")
        st.metric("Probability DSCR < 1.2", f"{breaches / max(1, int(n_sims)):.1%}")

    # Now render all charts together (grouped)
    st.markdown("### Figures")
    fig_cols = st.columns(1)
    with fig_cols[0]:
        if wf is not None:
            st.plotly_chart(wf, use_container_width=True)
        if fig_monte is not None:
            st.plotly_chart(fig_monte, use_container_width=True)

    # Export Monte Carlo chart for PDF as well
    if fig_monte is not None:
        png_mc = export_plotly_to_png_bytes(fig_monte)
        if png_mc:
            png_images.append((png_mc, "LP IRR Distribution (Monte Carlo)"))

    # Add any other chart exports you want (e.g., GPR/NOI charts) similar to above

    # Build downloadables (CSV + PDFs)
    csv_bytes = det["cf_table"].to_csv(index=False).encode()
    st.download_button("Download deterministic CF CSV", csv_bytes, "deterministic_cf.csv", "text/csv")

    if irrs.size > 0:
        buf = io.StringIO()
        pd.DataFrame({"LP_IRR": irrs}).to_csv(buf, index=False)
        st.download_button("Download Monte Carlo IRRs (CSV)", buf.getvalue().encode(), "mc_irrs.csv", "text/csv")

    # Prepare PDF memo (include PNG images list)
    monte_stats = {
        "p5": f"{p5:.1%}" if (p5 is not None) else "N/A",
        "p50": f"{p50:.1%}" if (p50 is not None) else "N/A",
        "p95": f"{p95:.1%}" if (p95 is not None) else "N/A",
    }

    # Collect logo bytes (if provided)
    logo_blob = None
    if logo_file:
        try:
            logo_blob = (logo_file.getvalue(), "uploaded")
        except Exception:
            logo_blob = None
    elif logo_url:
        try:
            r = requests.get(logo_url, timeout=8)
            if r.status_code == 200:
                logo_blob = (r.content, "url")
        except Exception:
            logo_blob = None

    with st.spinner("Generating PDF memo (multi-page) ..."):
        try:
            pdf_bytes, filename = generate_long_pdf_memo(det, monte_stats, png_images, logo_blob)
            mime = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
            st.success("PDF memo ready")
            st.download_button("Download Full PDF Memo", pdf_bytes, filename, mime)
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            summary_text = (f"Deterministic LP IRR: {lp_irr:.2%}\n"
                            f"P50 (MC): {monte_stats.get('p50')}\n"
                            f"P95 (MC): {monte_stats.get('p95')}\n")
            st.download_button("Download Summary (TXT)", summary_text.encode(), "proforma_summary.txt", "text/plain")

st.markdown("---")
st.info("This is an institutional-grade model with multi-tier promote settlement, Monte Carlo analysis, and automated multi-page PDF reporting (with logo/header/footer). Adjust inputs and rerun.")
