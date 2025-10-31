# app.py
"""
Robust Streamlit Global Currency Converter + Visual Analytics Dashboard
Uses exchangerate.host (free, no API key).
If the API fails, the app uses a small fallback so the UI remains usable.
Save this file as app.py and run: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import re
import time

# ---------------------
# Page config & CSS
# ---------------------
st.set_page_config(page_title="Global Currency Converter · Luxe", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#061224 0%, #001219 60%); color: #e6eef8; font-family: Inter, sans-serif; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:14px; margin-bottom:12px; }
    .title { font-weight:700; }
    .muted { color: #9fb3c8; }
    .stButton>button { background: linear-gradient(90deg,#7c3aed,#06b6d4); color: white; border-radius:10px; padding:8px 12px; }
    input, select, textarea { background: rgba(255,255,255,0.02) !important; color: #e6eef8 !important; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------
# Constants & fallbacks
# ---------------------
API_BASE = "https://api.exchangerate.host"
TOP10 = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "SEK", "NZD"]

CURRENCY_SYMBOLS = {
    "USD":"$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥","AUD":"A$","CAD":"C$",
    "CHF":"CHF","SEK":"kr","NZD":"NZ$","PKR":"₨","INR":"₹"
}

# small fallback dataset so the UI stays functional offline
FALLBACK_RATES_USD = {
    "USD":1.0, "EUR":0.92, "JPY":150.0, "GBP":0.79, "AUD":1.50, "CAD":1.34,
    "CHF":0.92, "CNY":7.20, "SEK":11.0, "NZD":1.61, "PKR":285.0, "INR":83.0
}

# ---------------------
# Utility functions
# ---------------------
def safe_request_json(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # bubble up exception text for UI display
        raise RuntimeError(f"Network/API error: {e}")

@st.cache_data(ttl=300)
def fetch_symbols_cached():
    """Return mapping of symbol->info or fallback minimal list if API unreachable."""
    try:
        data = safe_request_json(f"{API_BASE}/symbols")
        syms = data.get("symbols", {})
        if not syms:
            raise RuntimeError("Symbols empty from API.")
        return syms
    except Exception as e:
        # fallback create limited symbols from fallback rates
        fallback = {k: {"description": k, "code": k} for k in FALLBACK_RATES_USD.keys()}
        return fallback

def fetch_latest_rates(base: str="USD", force_refresh: bool=False):
    """
    Fetch latest rates for `base`.
    If force_refresh is True, bypass cache by adding unique param; caching still applies across identical calls.
    """
    params = {"base": base}
    if force_refresh:
        params["_"] = str(time.time())
    try:
        data = safe_request_json(f"{API_BASE}/latest", params=params)
        rates = data.get("rates")
        if not rates:
            raise RuntimeError("API returned no rates.")
        return {"rates": rates, "base": data.get("base", base), "date": data.get("date", "")}
    except Exception as e:
        # Provide a sensible fallback (convert by USD fallback)
        # If base==USD, we can use FALLBACK_RATES_USD directly
        if base.upper() == "USD":
            return {"rates": FALLBACK_RATES_USD, "base": "USD", "date": "fallback"}
        else:
            # try to compute rates relative to base using fallback: rate(target) = fallback[target] / fallback[base]
            base_upper = base.upper()
            if base_upper in FALLBACK_RATES_USD:
                derived = {}
                for k,v in FALLBACK_RATES_USD.items():
                    derived[k] = v / FALLBACK_RATES_USD[base_upper]
                return {"rates": derived, "base": base_upper, "date": "fallback-derived"}
        raise RuntimeError(f"Failed to fetch rates and no fallback available: {e}")

@st.cache_data(ttl=600)
def fetch_time_series_cached(base: str, target: str, days: int=30, force_refresh: bool=False):
    """Fetch historical timeseries. On error, produce synthetic series from fallback (so chart still shows)."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "base": base, "symbols": target}
    if force_refresh:
        params["_"] = str(time.time())
    try:
        data = safe_request_json(f"{API_BASE}/timeseries", params=params, timeout=20)
        if not data.get("success", False):
            raise RuntimeError("Timeseries API returned unsuccessful")
        rates = data.get("rates", {})
        df = pd.DataFrame([{"date": pd.to_datetime(d), "rate": day.get(target)} for d, day in rates.items()]).sort_values("date")
        return df
    except Exception:
        # Synthetic simple timeseries: use latest rate and add small noise
        try:
            latest = fetch_latest_rates(base)
            rate = latest["rates"].get(target)
            if rate is None:
                rate = 1.0
        except Exception:
            rate = 1.0
        dates = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame({"date": dates, "rate": [rate * (1 + 0.005 * (i % 5 - 2)) for i in range(len(dates))]})
        return df

# ---------------------
# NLP parse
# ---------------------
AMOUNT_CURRENCY_RE = re.compile(r"(?P<amt>[\d,.]+)\s*(?P<base>[A-Za-z]{3})\s*(?:to|in|into|->)?\s*(?P<target>[A-Za-z]{3})", re.IGNORECASE)
def parse_nl_input(text: str):
    if not text:
        return None
    m = AMOUNT_CURRENCY_RE.search(text)
    if m:
        amt = m.group("amt").replace(",", "")
        try:
            amount = float(amt)
        except:
            amount = None
        return amount, m.group("base").upper(), m.group("target").upper()
    # fallback simple parse: number + two codes
    nums = re.findall(r"[\d,.]+", text)
    codes = re.findall(r"\b[A-Za-z]{3}\b", text)
    if nums and len(codes) >= 2:
        try:
            amount = float(nums[0].replace(",", ""))
            return amount, codes[0].upper(), codes[1].upper()
        except:
            return None
    return None

# ---------------------
# Conversion core
# ---------------------
def convert_currency(amount: float, base: str, target: str, force_refresh: bool=False):
    base = base.upper(); target = target.upper()
    data = fetch_latest_rates(base=base, force_refresh=force_refresh)
    rates = data.get("rates", {})
    if target not in rates:
        raise ValueError(f"Target currency '{target}' not available (base {base}).")
    rate = rates[target]
    return {"amount": amount, "base": base, "target": target, "rate": rate, "converted": amount * rate, "date": data.get("date")}

# ---------------------
# UI - Sidebar
# ---------------------
st.sidebar.markdown("<div class='card'><h3 class='title'>✨ Luxe Currency Dashboard</h3><p class='muted'>Real-time conversions · No API key</p></div>", unsafe_allow_html=True)
symbols = fetch_symbols_cached()
CURRENCY_LIST = sorted(symbols.keys())

theme = st.sidebar.radio("Theme", ["Dark (Recommended)", "Light"])
if theme == "Light":
    st.markdown("<style>.stApp{background:linear-gradient(90deg,#f6f9fc,#ffffff); color:#0b2545}</style>", unsafe_allow_html=True)

force_refresh_flag = st.sidebar.button("Force refresh rates now (bypass cache)")

# ---------------------
# Main UI
# ---------------------
col1, col2 = st.columns([1.4, 2.6])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Convert currency  🔁")
    nl_input = st.text_input("Enter (e.g. 'convert 500 USD to PKR')", placeholder="convert 100 USD to EUR")
    parsed = parse_nl_input(nl_input) if nl_input else None

    # manual inputs (use parsed to prefill if valid)
    default_amount = float(parsed[0]) if parsed and parsed[0] is not None else 100.0
    amount_input = st.number_input("Amount", min_value=0.0, value=default_amount, format="%.2f")
    # ensure some defaults exist in currency list
    def idx_of(code, fallback):
        try:
            return CURRENCY_LIST.index(code)
        except Exception:
            try:
                return CURRENCY_LIST.index(fallback)
            except Exception:
                return 0
    base_default = parsed[1] if parsed and parsed[1] in CURRENCY_LIST else "USD"
    target_default = parsed[2] if parsed and parsed[2] in CURRENCY_LIST else ("PKR" if "PKR" in CURRENCY_LIST else ("EUR" if "EUR" in CURRENCY_LIST else CURRENCY_LIST[0]))
    base_currency = st.selectbox("Base currency", CURRENCY_LIST, index=idx_of(base_default, "USD"))
    target_currency = st.selectbox("Target currency", CURRENCY_LIST, index=idx_of(target_default, "USD"))
    convert_btn = st.button("Convert")
    st.markdown("---")
    st.markdown("**Quick examples**: `convert 2500 EUR to USD`, `5000 GBP in PKR`, `100 JPY to USD`")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Conversion result")
    do_convert = convert_btn or (parsed is not None and nl_input.strip() != "")
    try:
        if do_convert:
            # If parsed input provided valid currency codes, override manual controls
            if parsed and parsed[1] in CURRENCY_LIST and parsed[2] in CURRENCY_LIST and parsed[0] is not None:
                amount_input, base_currency, target_currency = parsed[0], parsed[1], parsed[2]
            # Perform conversion (respect force_refresh_flag)
            result = convert_currency(amount_input, base_currency, target_currency, force_refresh=force_refresh_flag)
            sym = CURRENCY_SYMBOLS.get(target_currency, target_currency)
            st.markdown(f"<div style='display:flex;gap:14px;align-items:center'><div style='font-size:20px;font-weight:700'>{amount_input:,.2f} {base_currency} →</div><div style='font-size:34px;font-weight:800;color:#9be7ff'>{sym} {result['converted']:,.4f} <span style='font-size:14px;color:#a8cfe0'>({target_currency})</span></div></div>", unsafe_allow_html=True)
            st.write(f"Rate: 1 {base_currency} = {result['rate']:.6f} {target_currency}  •  Source date: {result['date']}")
            st.success("Conversion successful")
        else:
            st.info("Enter values and click Convert, or use natural-language input.")
    except Exception as e:
        st.error(f"Conversion error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
# Charts row
# ---------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("Visual Analytics")

chart_col1, chart_col2 = st.columns([1.6, 1.4])
latest_data = None
try:
    latest_data = fetch_latest_rates(base_currency, force_refresh=force_refresh_flag)
    rates = latest_data.get("rates", {})
except Exception as e:
    st.warning(f"Could not fetch fresh rates from API: {e}. Using fallback rates where possible.")
    rates = FALLBACK_RATES_USD

with chart_col1:
    st.subheader("Top 10 currency comparison")
    try:
        comparison = []
        for c in TOP10:
            if c == base_currency:
                comparison.append({"currency": c, "rate": 1.0})
            else:
                comparison.append({"currency": c, "rate": rates.get(c)})
        df_comp = pd.DataFrame(comparison).dropna().sort_values("rate", ascending=False)
        fig_bar = px.bar(df_comp, x="currency", y="rate", text="rate", title=f"Rates relative to {base_currency}", template="plotly_dark")
        fig_bar.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=380)
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Comparison chart error: {e}")

with chart_col2:
    st.subheader("Distribution of converted value (top 6)")
    try:
        top6 = TOP10[:6]
        dist = []
        for c in top6:
            r = 1.0 if c == base_currency else rates.get(c)
            if r:
                dist.append({"currency": c, "value": amount_input * r})
        df_dist = pd.DataFrame(dist)
        fig_pie = px.pie(df_dist, names="currency", values="value", title=f"Converted value distribution ({amount_input} {base_currency})", template="plotly_dark")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    except Exception as e:
        st.error(f"Distribution chart error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
# Historical trend
# ---------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("Historical trend for selected pair")
ts_col1, ts_col2 = st.columns([2,1])
with ts_col1:
    days = st.select_slider("Days to show", options=[7,14,30,60,90], value=30)
    try:
        df_ts = fetch_time_series_cached(base_currency, target_currency, days=days, force_refresh=force_refresh_flag)
        if df_ts.empty:
            st.warning("No historical data available; showing synthetic series.")
        fig_line = px.line(df_ts, x="date", y="rate", title=f"{base_currency}/{target_currency} — last {days} days", template="plotly_dark", markers=True)
        fig_line.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_line, use_container_width=True)
    except Exception as e:
        st.error(f"Historical chart error: {e}")

with ts_col2:
    st.subheader("Stats")
    try:
        if 'df_ts' in locals() and not df_ts.empty:
            low = df_ts['rate'].min(); high = df_ts['rate'].max(); last = df_ts['rate'].iloc[-1]
            pct_change = ((last - df_ts['rate'].iloc[0]) / df_ts['rate'].iloc[0]) * 100 if df_ts['rate'].iloc[0] != 0 else 0
            st.metric(label="Latest rate", value=f"{last:.6f}", delta=f"{pct_change:.2f}%")
            st.write(f"Lowest (last {days} days): {low:.6f}")
            st.write(f"Highest (last {days} days): {high:.6f}")
        else:
            st.write("No timeseries to calculate stats.")
    except Exception as e:
        st.error(f"Stats error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------
# Footer + tips
# ---------------------
st.markdown("<div class='card'><small class='muted'>Data service: exchangerate.host (free, no API key). If the API is unreachable the app uses a fallback dataset so visuals still render.</small></div>", unsafe_allow_html=True)

st.markdown("""
**Troubleshooting tips**
- If the app shows network errors, check your PC/VM network and that https://api.exchangerate.host is reachable from your network.
- If packages fail to import, run: `python -m pip install -r requirements.txt` then restart the streamlit process.
- If currency dropdown is empty, there's an issue fetching symbols — the app will still populate with fallback currencies.
""")
