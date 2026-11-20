import json
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# 示範觀察名單（可自由擴充至完整港股通名單，4位數字補零 + .HK）
WATCHLIST = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "9999.HK", "1211.HK",
    "0981.HK", "2318.HK", "0388.HK", "0005.HK", "1299.HK", "0883.HK", "0941.HK",
    "2899.HK", "1024.HK", "2015.HK", "9868.HK", "0285.HK", "2382.HK", "9992.HK",
    "1133.HK", "1686.HK", "9660.HK", "2858.HK", "0020.HK", "6690.HK", "0175.HK",
    "2269.HK", "1093.HK", "1177.HK", "0992.HK", "0763.HK", "1347.HK", "6160.HK"
]

def get_hsi_market_status():
    """計算 M (Market Direction)：恒指趨勢與近 25 日派發日數"""
    hsi = yf.download("^HSI", period="1y", progress=False, auto_adjust=True)
    if isinstance(hsi.columns, pd.MultiIndex):
        hsi.columns = hsi.columns.get_level_values(0)
    hsi = hsi.dropna()

    close = hsi["Close"]
    vol = hsi["Volume"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last_close = close.iloc[-1]

    # 派發日：跌幅 >= 0.2% 且成交量高於前一日（近 25 個交易日）
    pct = close.pct_change()
    dist_days = ((pct <= -0.002) & (vol > vol.shift(1))).tail(25).sum()

    if last_close > sma50 and last_close > sma200 and dist_days <= 4:
        status = "確認上升趨勢 (Confirmed Uptrend)"
        signal = "GREEN"
    elif last_close > sma200:
        status = "上升趨勢受壓 (Uptrend Under Pressure)"
        signal = "YELLOW"
    else:
        status = "大市調整中 (Market in Correction)"
        signal = "RED"

    hk_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M HKT")
    return {
        "updated_at": hk_time,
        "hsi_close": round(float(last_close), 2),
        "sma50": round(float(sma50), 2),
        "sma200": round(float(sma200), 2),
        "dist_days_25d": int(dist_days),
        "status": status,
        "signal": signal
    }

def run_scan():
    market_info = get_hsi_market_status()
    with open("market_status.json", "w", encoding="utf-8") as f:
        json.dump(market_info, f, ensure_ascii=False, indent=2)

    # 批次下載 1 年日線數據
    raw = yf.download(WATCHLIST, period="1y", group_by="ticker", progress=False, auto_adjust=True)
    records = []

    for ticker in WATCHLIST:
        try:
            df = raw[ticker].dropna() if len(WATCHLIST) > 1 else raw.dropna()
            if len(df) < 200:
                continue

            close = df["Close"]
            vol = df["Volume"]
            last_close = float(close.iloc[-1])

            # S: 流動性（20日均成交額，百萬港元）
            avg_turnover_m = float((close * vol).tail(20).mean() / 1e6)

            # L & N: 均線趨勢與 52 週新高距離
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma150 = float(close.rolling(150).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            high_52w = float(df["High"].max())
            dist_52w_pct = (last_close / high_52w - 1) * 100

            trend_ok = (last_close > sma50 > sma150 > sma200)

            # IBD 風格加權相對強度原始分 (3M*40% + 6M*20% + 9M*20% + 12M*20%)
            r3 = close.iloc[-1] / close.iloc[-63] - 1 if len(close) > 63 else 0
            r6 = close.iloc[-1] / close.iloc[-126] - 1 if len(close) > 126 else 0
            r9 = close.iloc[-1] / close.iloc[-189] - 1 if len(close) > 189 else 0
            r12 = close.iloc[-1] / close.iloc[0] - 1
            rs_raw = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12

            # 量比 (RVOL: 當日量 / 50日均量)
            rvol = float(vol.iloc[-1] / vol.rolling(50).mean().iloc[-1])

            records.append({
                "Ticker": ticker,
                "Price": round(last_close, 2),
                "Turnover_20D_M": round(avg_turnover_m, 1),
                "Dist_52W_High_%": round(dist_52w_pct, 1),
                "Trend_Template": trend_ok,
                "RVOL": round(rvol, 2),
                "RS_Raw": rs_raw
            })
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return

    # 計算相對強度百分位 RS Rating (1 - 99)
    res_df["RS_Rating"] = (res_df["RS_Raw"].rank(pct=True) * 99).round(0).astype(int)

    # C & A: 抓取基本面（僅針對成交額 >= 2,000萬且股價 >= $2 的股份）
    names, eps_g, rev_g, roes, mcaps = [], [], [], [], []
    for _, row in res_df.iterrows():
        t = row["Ticker"]
        if row["Turnover_20D_M"] >= 20 and row["Price"] >= 2.0:
            try:
                info = yf.Ticker(t).info
                names.append(info.get("shortName", t))
                eps_g.append(round(info["earningsGrowth"] * 100, 1) if info.get("earningsGrowth") is not None else np.nan)
                rev_g.append(round(info["revenueGrowth"] * 100, 1) if info.get("revenueGrowth") is not None else np.nan)
                roes.append(round(info["returnOnEquity"] * 100, 1) if info.get("returnOnEquity") is not None else np.nan)
                mcaps.append(round(info["marketCap"] / 1e9, 1) if info.get("marketCap") else np.nan)
                time.sleep(0.3)  # 避免請求過快
            except Exception:
                names.append(t); eps_g.append(np.nan); rev_g.append(np.nan); roes.append(np.nan); mcaps.append(np.nan)
        else:
            names.append(t); eps_g.append(np.nan); rev_g.append(np.nan); roes.append(np.nan); mcaps.append(np.nan)

    res_df["Name"] = names
    res_df["MktCap_B"] = mcaps
    res_df["EPS_Growth_%"] = eps_g
    res_df["Rev_Growth_%"] = rev_g
    res_df["ROE_%"] = roes

    # 排序並輸出
    cols = ["Ticker", "Name", "Price", "MktCap_B", "Turnover_20D_M", "RS_Rating",
            "Dist_52W_High_%", "Trend_Template", "RVOL", "EPS_Growth_%", "Rev_Growth_%", "ROE_%"]
    res_df = res_df[cols].sort_values(by="RS_Rating", ascending=False)
    res_df.to_csv("canslim_hk.csv", index=False, encoding="utf-8-sig")
    print("Scan completed successfully.")

if __name__ == "__main__":
    run_scan()
