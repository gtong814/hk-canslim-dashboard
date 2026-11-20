import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="港股 CAN SLIM Dashboard", layout="wide")
st.title("🇭🇰 港股 CAN SLIM 量化選股 Dashboard")

# 1. 讀取 M (大市方向) 狀態
try:
    with open("market_status.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    st.caption(f"最後更新：{m['updated_at']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("恒生指數 (HSI)", f"{m['hsi_close']:,.2f}")
    c2.metric("對比 50 天線", f"{m['sma50']:,.2f}", delta=f"{m['hsi_close'] - m['sma50']:+.1f}")
    c3.metric("近 25 日派發日", f"{m['dist_days_25d']} 日")
    if m["signal"] == "GREEN":
        c4.success(f"M 狀態：{m['status']}")
    elif m["signal"] == "YELLOW":
        c4.warning(f"M 狀態：{m['status']}")
    else:
        c4.error(f"M 狀態：{m['status']}")
except FileNotFoundError:
    st.warning("尚未生成市場數據，請先到 GitHub Actions 手動執行一次 Daily HK CAN SLIM Scan。")
    st.stop()

# 2. 讀取選股結果
df = pd.read_csv("canslim_hk.csv")

# 側邊欄篩選器
st.sidebar.header("CAN SLIM 篩選條件")
min_rs = st.sidebar.slider("L — 最低 RS Rating", 0, 99, 70)
min_turnover = st.sidebar.slider("S — 最低 20 日均成交額 (百萬港元)", 0, 500, 50)
max_dist_high = st.sidebar.slider("N — 距 52 週高位以內 (%)", -50, 0, -25)
only_trend = st.sidebar.checkbox("只顯示多頭排列 (價 > 50 > 150 > 200)", value=False)
use_fund = st.sidebar.checkbox("啟用 C/A 基本面過濾 (EPS/ROE)", value=False)

filtered = df[
    (df["RS_Rating"] >= min_rs) &
    (df["Turnover_20D_M"] >= min_turnover) &
    (df["Dist_52W_High_%"] >= max_dist_high)
]

if only_trend:
    filtered = filtered[filtered["Trend_Template"] == True]
if use_fund:
    min_eps = st.sidebar.number_input("C — 最低 EPS 增長 (%)", value=20.0)
    min_roe = st.sidebar.number_input("A — 最低 ROE (%)", value=15.0)
    filtered = filtered[
        (filtered["EPS_Growth_%"] >= min_eps) &
        (filtered["ROE_%"] >= min_roe)
    ]

# 3. 顯示篩選表格
st.subheader(f"符合條件股份 ({len(filtered)} 隻)")
st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "RS_Rating": st.column_config.ProgressColumn("RS 強度", min_value=0, max_value=99, format="%d"),
        "Turnover_20D_M": st.column_config.NumberColumn("20日均成交(M)", format="%.1f"),
        "Dist_52W_High_%": st.column_config.NumberColumn("距52週高(%)", format="%.1f%%"),
    }
)

# 4. 互動個股圖表 (K線 + 50/150/200 天線)
if not filtered.empty:
    st.divider()
    selected = st.selectbox("選擇股份檢視技術走勢：", filtered["Ticker"].tolist())

    @st.cache_data(ttl=3600)
    def load_chart_data(ticker):
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA150"] = data["Close"].rolling(150).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()
        return data

    cdf = load_chart_data(selected)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=cdf.index, open=cdf["Open"], high=cdf["High"],
        low=cdf["Low"], close=cdf["Close"], name="K線"
    ))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA50"], line=dict(color="orange", width=1.5), name="SMA 50"))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA150"], line=dict(color="blue", width=1.5), name="SMA 150"))
    fig.add_trace(go.Scatter(x=cdf.index, y=cdf["SMA200"], line=dict(color="purple", width=1.8), name="SMA 200"))

    fig.update_layout(
        title=f"{selected} 日線圖與均線結構",
        xaxis_rangeslider_visible=False,
        height=520,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
