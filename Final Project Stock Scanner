import pandas as pd
import yfinance as yf
import time
import requests
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------- SETTINGS ----------------
SOURCES = [
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nyse",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=amex"
]
KEYWORDS = ["bio", "pharma", "thera", "diagnostic", "gen", "immune", "onc"]
SLEEP = 0.05
MIN_PRICE = 0.20
MAX_PRICE = 10
MIN_VOLUME = 200000

# ---------------- TICKER FETCH (LIVE ONLY) ----------------
def fetch_biotech():
    tickers = set()
    for url in tqdm(SOURCES, desc="Fetching exchanges"):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            js = r.json()
            rows = js.get("data", {}).get("table", {}).get("rows", [])
            exchange = url.split("exchange=")[-1].upper()
            print(f"{exchange}: {len(rows)} rows")
            for row in rows:
                sym = row.get("symbol")
                name = row.get("name", "").lower()
                if sym and any(k in name for k in KEYWORDS):
                    tickers.add(sym)
        except Exception as e:
            print(f"Error fetching {url.split('=')[-1]}: {e}")
    
    final = sorted(list(tickers))
    print(f"\nTotal biotech tickers: {len(final)} (live only)")
    return final

# ---------------- STOCK SCORING ----------------
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
        score = pct_from_low * 0.2 + vol_trend * 50
        return {
            "ticker": ticker,
            "price": round(price, 3),
            "vol": int(vol),
            "% from low": round(pct_from_low, 1),
            "vol trend": round(vol_trend, 2),
            "score": round(score, 2)
        }
    except Exception:
        return None

# ---------------- FEATURE ENGINEERING ----------------
def add_features(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", auto_adjust=True)
        if len(hist) < 40:
            return None
        price_change_1m = (hist["Close"].iloc[-1] - hist["Close"].iloc[-20]) / hist["Close"].iloc[-20]
        volatility = hist["Close"].iloc[-20:].std()
        return round(price_change_1m, 3), round(volatility, 3)
    except Exception:
        return None

# ---------------- LABELING ----------------
def label_stock(ticker):
    try:
        d = yf.Ticker(ticker)
        hist = d.history(period="3mo", auto_adjust=True)
        if len(hist) < 40:
            return None
        price_20d_ago = hist["Close"].iloc[-21]  # 20 trading days ago
        price_today = hist["Close"].iloc[-1]
        future_return = (price_today - price_20d_ago) / price_20d_ago
        return 1 if future_return > 0.3 else 0
    except Exception:
        return None

# ---------------- MAIN PIPELINE ----------------
def main():
    tickers = fetch_biotech()
    print("Sample tickers:", tickers[:15])

    scored = []
    for t in tqdm(tickers, desc="Scoring stocks"):
        time.sleep(SLEEP)
        result = score_stock(t)
        if result:
            scored.append(result)

    df = pd.DataFrame(scored)
    print(f"Scored: {len(df)}")

    # Add features
    df["features"] = df["ticker"].apply(add_features)
    df = df.dropna(subset=["features"]).copy()
    df[["price_change_1m", "volatility"]] = pd.DataFrame(df["features"].tolist(), index=df.index)
    df.drop(columns=["features"], inplace=True)

    # Add labels
    df["label"] = df["ticker"].apply(label_stock)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Prepare ML
    X = df[["price_change_1m", "volatility", "% from low", "vol trend"]]
    y = df["label"]

    if len(X) < 10:
        print("Not enough data for ML.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df["predicted_prob"] = model.predict_proba(X)[:, 1]
    df["predicted_label"] = model.predict(X)

    top = df.sort_values("predicted_prob", ascending=False).head(20)
    print("\nTop 20 Biotech Picks:")
    print(top[["ticker", "price", "vol", "% from low", "vol trend", "price_change_1m", "volatility", "predicted_prob"]].to_string(index=False))

if __name__ == "__main__":
    main()
