import pandas as pd
import yfinance as yf
import time
import requests
from tqdm import tqdm
from xgboost import XGBClassifier
import numpy as np
from datetime import datetime

# ---------------- SETTINGS ----------------
SLEEP = 0.2
MIN_PRICE = 0.20
MAX_PRICE = 15
MIN_VOLUME = 100000
MIN_MARKET_CAP = 10_000_000
MAX_MARKET_CAP = 500_000_000

# YOUR ORIGINAL SOURCES
SOURCES = [
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nyse",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=amex"
]
KEYWORDS = ["bio", "pharma", "thera", "diagnostic", "gen", "immune", "onc", "neuro", "psych", "cns"]

# Auto CSV
TODAY = datetime.now().strftime("%b%d_%Y")
CSV_FILE = f"NRXP_clones_HYBRID_{TODAY}.csv"

# ---------------- 1. SCRAPE ALL EXCHANGES + KEYWORDS ----------------
def fetch_keyword_biotechs():
    tickers = set()
    for url in tqdm(SOURCES, desc="Scraping exchanges"):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            js = r.json()
            rows = js.get("data", {}).get("table", {}).get("rows", [])
            for row in rows:
                sym = row.get("symbol")
                name = row.get("name", "").lower()
                if sym and any(k in name for k in KEYWORDS):
                    tickers.add(sym)
        except Exception as e:
            print(f"Error scraping {url.split('=')[-1]}: {e}")
        time.sleep(0.1)
    print(f"Biotech keyword tickers: {len(tickers)}")
    return sorted(list(tickers))

# ---------------- 2. GET PHASE 3 CNS SPONSORS ----------------
def get_phase3_cns_sponsors():
    base = "https://clinicaltrials.gov/api/v2/studies"
    expr = (
        'AREA[Phase]Phase 3 AND AREA[StudyType]Interventional AND AREA[InterventionType]Drug '
        'AND (AREA[ConditionSearch](depression OR bipolar OR suicide OR PTSD OR schizophrenia OR anxiety OR addiction OR Alzheimer OR Parkinson OR neurology OR psychiatry)) '
        'AND AREA[Status]NOT (Terminated OR Withdrawn OR Suspended)'
    )
    url = f"{base}?format=json&query.expr={expr}&pageSize=1000"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
    except Exception as e:
        print("ClinicalTrials.gov failed:", e)
        return set()

    sponsors = set()
    for study in data.get("studies", []):
        name = study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"]
        clean = name.upper().split(" INC")[0].split(" CORPORATION")[0].split(" LTD")[0].split(",")[0].strip()
        sponsors.add(clean)
    print(f"Phase 3 CNS sponsors: {len(sponsors)}")
    return sponsors

# ---------------- 3. HYBRID MATCHING ----------------
def hybrid_filter(keyword_tickers, phase3_sponsors):
    print("Matching company names...")
    ticker_to_name = {}
    for t in tqdm(keyword_tickers, desc="Fetching company names"):
        try:
            info = yf.Ticker(t).info
            name = info.get("longName", "") or info.get("shortName", "")
            if name:
                ticker_to_name[t] = name.upper()
        except:
            pass
        time.sleep(0.05)

    matches = set()
    for ticker, full_name in ticker_to_name.items():
        for sponsor in phase3_sponsors:
            if sponsor in full_name or any(word in full_name for word in sponsor.split()):
                matches.add(ticker)
                break

    # Hardcode known NRXP-style tickers in case of name mismatch
    known_good = {"NRXP", "VTGN", "RLMD", "GHRS", "CYBN", "AXSM", "NMRA", "ATYR", "ALTO", "SILO", "LEXX"}
    matches.update(known_good)

    # Final filter by size/price/volume
    final = []
    for t in matches:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap", 0)
            price = info.get("currentPrice") or info.get("regularMarketPreviousClose") or 0
            vol = info.get("averageVolume", 0) or 0
            if (MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP and
                MIN_PRICE <= price <= MAX_PRICE and
                vol >= MIN_VOLUME):
                final.append(t)
        except:
            pass
        time.sleep(0.05)

    print(f"Final hybrid NRXP-clones: {len(final)}")
    return sorted(final)

# ---------------- SCORING ----------------
def score_stock(ticker):
    try:
        d = yf.Ticker(ticker)
        hist = d.history(period="6mo", auto_adjust=True)
        if hist.empty or len(hist) < 20:
            return None
        price = hist["Close"].iloc[-1]
        vol = hist["Volume"].iloc[-20:].mean()
        if price < MIN_PRICE or price > MAX_PRICE or vol < MIN_VOLUME:
            return None
        low = hist["Close"].min()
        pct_from_low = (price - low) / low * 100
        vol_trend = (hist["Volume"].iloc[-1] - hist["Volume"].iloc[0]) / max(hist["Volume"].iloc[0], 1)
        return {
            "ticker": ticker,
            "price": round(price, 3),
            "vol": int(vol),
            "% from low": round(pct_from_low, 1),
            "vol trend": round(vol_trend, 2)
        }
    except:
        return None

# ---------------- FEATURES ----------------
def add_features(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        info = yf.Ticker(ticker).info
        if len(hist) < 100:
            return None
        close = hist["Close"]
        vol = hist["Volume"]
        pct_from_low = (close.iloc[-1] / close.min()) - 1
        days_since_low = (hist.index[-1] - close.idxmin()).days
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(14).mean()
        ma_down = down.rolling(14).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        float_ratio = info.get("floatShares", 1) / max(info.get("sharesOutstanding", 1), 1)
        short_interest = info.get("shortPercentOfFloat", 0) or 0
        vol_spike = vol.iloc[-1] / vol.iloc[-20:].mean()
        return {
            "pct_from_low": round(pct_from_low, 3),
            "days_since_low": int(days_since_low),
            "rsi": round(rsi_val, 2),
            "float_ratio": round(float_ratio, 3),
            "short_interest": round(short_interest, 3),
            "vol_spike": round(vol_spike, 2)
        }
    except:
        return None

# ---------------- LABEL (2x in 60 days) ----------------
def label_stock(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if len(hist) < 120:
            return None
        prices = hist["Close"]
        entry = prices.iloc[-60]
        highest = prices.iloc[-60:].max()
        return 1 if (highest / entry >= 2.0) else 0
    except:
        return None

# ---------------- MAIN ----------------
def main():
    print("Starting HYBRID NRXP-clone scanner (Nasdaq screener + ClinicalTrials.gov)\n")
    
    keyword_tickers = fetch_keyword_biotechs()
    phase3_sponsors = get_phase3_cns_sponsors()
    tickers = hybrid_filter(keyword_tickers, phase3_sponsors)
    
    if len(tickers) < 5:
        print("Not enough matches today.")
        return

    # Scoring
    scored = []
    for t in tqdm(tickers, desc="Scoring"):
        time.sleep(SLEEP)
        res = score_stock(t)
        if res:
            scored.append(res)
    
    if not scored:
        print("No stocks passed scoring.")
        return
    
    df = pd.DataFrame(scored)
    print(f"Scored: {len(df)} tickers")

    # Features
    print("Adding features...")
    df["features"] = df["ticker"].apply(add_features)
    df = df.dropna(subset=["features"]).copy()
    if df.empty:
        print("No ticker had enough data for features.")
        return
    feature_df = pd.DataFrame(df["features"].tolist(), index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    df.drop(columns=["features"], inplace=True)

    # Labels
    print("Labeling moonshots...")
    df["label"] = df["ticker"].apply(label_stock)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    if len(df) < 8:
        print("Not enough labeled data for ML.")
        return

    # ML
    cols = ["pct_from_low", "days_since_low", "rsi", "float_ratio", "short_interest", "vol_spike"]
    X = df[cols]
    y = df["label"]
    model = XGBClassifier(n_estimators=300, max_depth=5, scale_pos_weight=4, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    df["predicted_prob"] = model.predict_proba(X)[:, 1]

    top = df.sort_values("predicted_prob", ascending=False).head(20)
    top.to_csv(CSV_FILE, index=False)
    print(f"\nResults saved to: {CSV_FILE}")

    print("\n" + "="*80)
    print(f"TOP 20 NRXP CLONES — {len(top)} FOUND ({TODAY})")
    print("="*80)
    print(top[["ticker", "price", "vol", "% from low", "vol_spike", "predicted_prob"]].round(3).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
