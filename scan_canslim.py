import json
import os
import time
from datetime import datetime, timezone, timedelta
import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_CACHE = "ggt_universe.csv"

def code_to_yf_ticker(code: str) -> str:
    """將 5 位港股代碼 (如 00700) 轉為 Yahoo Finance 格式 (0700.HK)"""
    clean = str(code).strip().split(".")[0]
    return f"{str(int(clean)).zfill(4)}.HK"

def get_ggt_universe() -> dict:
    """透過 akshare 抓取最新全量港股通名單，並自動儲存快取作備援"""
    try:
        df = ak.stock_hk_ggt_components_em()
        # 保留「代碼」與「名稱」並轉換為 yfinance 格式
        df["Ticker"] = df["代码"].apply(code_to_yf_ticker)
        df["Name"] = df["名称"].astype(str).str.strip()
        uni_df = df[["Ticker", "Name"]].drop_duplicates(subset=["Ticker"])
        uni_df.to_csv(UNIVERSE_CACHE, index=False, encoding="utf-8-sig")
        print(f"成功從 akshare 取得 {len(uni_df)} 隻港股通成份股。")
        return dict(zip(uni_df["Ticker"], uni_df["Name"]))
    except Exception as e:
        print(f"akshare 抓取失敗 ({e})，嘗試讀取快取名單...")
        if os.path.exists(UNIVERSE_CACHE):
            uni_df = pd.read_csv(UNIVERSE_CACHE)
            print(f"已從快取載入 {len(uni_df)} 隻港股通成份股。")
            return dict(zip(uni_df["Ticker"], uni_df["Name"]))
        raise RuntimeError("無法取得港股通名單，且本地無快取檔案。")

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
    # 1. 更新大市狀態 (M)
    market_info = get_hsi_market_status()
    with open("market_status.json", "w", encoding="utf-8") as f:
        json.dump(market_info, f, ensure_ascii=False, indent=2)

    # 2. 取得全量港股通名單
    name_map = get_ggt_universe()
    tickers = list(name_map.keys())

    # 3. 分批下載 1 年日線數據（每批 80 隻）
    records = []
    batch_size = 80
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"正在下載第 {i+1} - {min(i+batch_size, len(tickers))} 隻股票數據...")
        try:
            raw = yf.download(batch, period="1y", group_by="ticker", progress=False, auto_adjust=True, threads=True)
        except Exception as e:
            print(f"批次下載失敗: {e}")
            continue

        for ticker in batch:
            try:
                df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
                if len(df) < 120:
                    continue

                close = df["Close"]
                vol = df["Volume"]
                last_close = float(close.iloc[-1])
                if last_close <= 0:
                    continue

                # S: 流動性（20日均成交額，百萬港元）
                avg_turnover_m = float((close * vol).tail(20).mean() / 1e6)

                # L & N: 均線趨勢與 52 週新高距離
                sma50 = float(close.rolling(50).mean().iloc[-1])
                sma150 = float(close.rolling(150).mean().iloc[-1]) if len(close) >= 150 else np.nan
                sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
                high_52w = float(df["High"].max())
                dist_52w_pct = (last_close / high_52w - 1) * 100

                trend_ok = bool(
                    pd.notna(sma200) and (last_close > sma50 > sma150 > sma200)
                )

                # IBD 風格加權相對強度原始分
                r3 = close.iloc[-1] / close.iloc[-63] - 1 if len(close) > 63 else 0
                r6 = close.iloc[-1] / close.iloc[-126] - 1 if len(close) > 126 else r3
                r9 = close.iloc[-1] / close.iloc[-189] - 1 if len(close) > 189 else r6
                r12 = close.iloc[-1] / close.iloc[0] - 1
                rs_raw = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12

                # 量比 (RVOL)
                avg_vol_50 = float(vol.rolling(50).mean().iloc[-1])
                rvol = float(vol.iloc[-1] / avg_vol_50) if avg_vol_50 > 0 else 0.0

                records.append({
                    "Ticker": ticker,
                    "Name": name_map.get(ticker, ticker),
                    "Price": round(last_close, 2),
                    "Turnover_20D_M": round(avg_turnover_m, 1),
                    "Dist_52W_High_%": round(dist_52w_pct, 1),
                    "Trend_Template": trend_ok,
                    "RVOL": round(rvol, 2),
                    "RS_Raw": rs_raw
                })
            except Exception:
                continue
        time.sleep(1.0)

    res_df = pd.DataFrame(records)
    if res_df.empty:
        print("未取得有效股票數據。")
        return

    # 4. 計算全量港股通 RS Rating (1 - 99)
    res_df["RS_Rating"] = (res_df["RS_Raw"].rank(pct=True) * 99).round(0).astype(int)

    # 5. 漏斗第二層 (C & A)：只對強勢且具流動性的股份抓取基本面，節省時間並防封鎖
    eps_g, rev_g, roes, mcaps = [], [], [], []
    fund_count = 0
    for _, row in res_df.iterrows():
        t = row["Ticker"]
        if row["RS_Rating"] >= 70 and row["Turnover_20D_M"] >= 30 and row["Price"] >= 2.0:
            try:
                info = yf.Ticker(t).info
                eps_g.append(round(info["earningsGrowth"] * 100, 1) if info.get("earningsGrowth") is not None else np.nan)
                rev_g.append(round(info["revenueGrowth"] * 100, 1) if info.get("revenueGrowth") is not None else np.nan)
                roes.append(round(info["returnOnEquity"] * 100, 1) if info.get("returnOnEquity") is not None else np.nan)
                mcaps.append(round(info["marketCap"] / 1e9, 1) if info.get("marketCap") else np.nan)
                fund_count += 1
                time.sleep(0.25)
            except Exception:
                eps_g.append(np.nan); rev_g.append(np.nan); roes.append(np.nan); mcaps.append(np.nan)
        else:
            eps_g.append(np.nan); rev_g.append(np.nan); roes.append(np.nan); mcaps.append(np.nan)

    print(f"已完成 {fund_count} 隻強勢股的基本面數據抓取。")
    res_df["MktCap_B"] = mcaps
    res_df["EPS_Growth_%"] = eps_g
    res_df["Rev_Growth_%"] = rev_g
    res_df["ROE_%"] = roes

    # 6. 排序並輸出 CSV
    cols = ["Ticker", "Name", "Price", "MktCap_B", "Turnover_20D_M", "RS_Rating",
            "Dist_52W_High_%", "Trend_Template", "RVOL", "EPS_Growth_%", "Rev_Growth_%", "ROE_%"]
    res_df = res_df[cols].sort_values(by="RS_Rating", ascending=False)
    res_df.to_csv("canslim_hk.csv", index=False, encoding="utf-8-sig")
    print(f"掃描完成！共輸出 {len(res_df)} 隻港股通股份。")

if __name__ == "__main__":
    run_scan()
