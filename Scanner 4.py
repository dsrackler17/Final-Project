import pandas as pd
import yfinance as yf
import time
import requests
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# ---------------- SAFE ML IMPORTS ----------------
models_available = {}

# XGBoost
try:
    from xgboost import XGBClassifier
    models_available["1"] = ("XGBoost", XGBClassifier(
        n_estimators=400, max_depth=6, scale_pos_weight=4,
        random_state=42, eval_metric='logloss'
    ))
except ImportError:
    print("XGBoost not found → pip install xgboost")

# Random Forest (always available)
models_available["2"] = ("Random Forest", RandomForestClassifier(
    n_estimators=500, max_depth=8, class_weight='balanced',
    random_state=42, n_jobs=-1
))

# LightGBM
try:
    from lightgbm import LGBMClassifier
    models_available["3"] = ("LightGBM", LGBMClassifier(
        n_estimators=400, max_depth=7, scale_pos_weight=4,
        random_state=42, verbose=-1
    ))
except ImportError:
    print("LightGBM not found → pip install lightgbm")

# CatBoost
try:
    from catboost import CatBoostClassifier
    models_available["4"] = ("CatBoost", CatBoostClassifier(
        iterations=400, depth=7, scale_pos_weight=4,
        random_state=42, verbose=False
    ))
except ImportError:
    print("CatBoost not found → pip install catboost")

# Logistic Regression
models_available["5"] = ("Logistic Regression", LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
))

# ---------------- SETTINGS ----------------
SLEEP = 0.2
TODAY = datetime.now().strftime("%b%d_%Y")
CSV_FILE = f"NRXP_clones_HYBRID_NEW_{TODAY}.csv"

SOURCES = [
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nyse",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=amex"
]

KEYWORDS = [
    "bio", "pharma", "thera", "peutics", "labs", "tech", "gene", "cell", "neuro",
    "psych", "cns", "immune", "onco", "rare", "mental", "brain", "addict", "alzheimer",
    "parkinson", "schizo", "bipolar", "depress", "anxiety", "ptsd", "suicide"
]

KNOWN_MOONSHOTS = {"NRXP", "VTGN", "RLMD", "GHRS", "CYBN", "AXSM", "NMRA", "ATYR", "ALTO", "SILO", "LEXX"}

# ---------------- MODEL SELECTION ----------------
def select_model():
    print("\n" + "="*60)
    print(" SELECT ML MODEL FOR NRXP-CLONE PREDICTION")
    print("="*60)

    if not models_available:
        print("ERROR: No ML models available!")
        return None, "None"

    for key in sorted(models_available.keys()):
        name, _ = models_available[key]
        print(f"{key}. {name}")

    boosters = [k for k in ["1", "3"] if k in models_available]
    can_ensemble = len(boosters) >= 2 or ("1" in models_available and "2" in models_available)
    if can_ensemble:
        print("6. Voting Ensemble (best available models)")

    while True:
        max_choice = len(models_available) + (1 if can_ensemble else 0)
        prompt = f"\nEnter your choice (1-{max_choice}) [default: 2]: "
        choice = input(prompt).strip() or "2"

        if choice in models_available:
            name, model = models_available[choice]
            print(f"Selected model: {name}\n")
            return model, name

        if can_ensemble and choice == "6":
            print("Building Voting Ensemble...")
            estimators = []
            if "1" in models_available:
                estimators.append(('xgb', XGBClassifier(n_estimators=300, max_depth=6, scale_pos_weight=4, random_state=42, eval_metric='logloss')))
            if "3" in models_available:
                estimators.append(('lgb', LGBMClassifier(n_estimators=300, max_depth=7, scale_pos_weight=4, random_state=42, verbose=-1)))
            if "2" in models_available:
                estimators.append(('rf', RandomForestClassifier(n_estimators=400, max_depth=8, class_weight='balanced', random_state=42)))
            ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            ens_name = f"Voting Ensemble ({len(estimators)} models)"
            print(f"Selected model: {ens_name}\n")
            return ensemble, ens_name

        print("Invalid choice, try again.")

# ---------------- DATA FETCHING ----------------
def fetch_keyword_biotechs():
    tickers = set()
    for url in tqdm(SOURCES, desc="Scraping exchanges"):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
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
    print(f"Biotech keyword tickers found: {len(tickers)}")
    return sorted(list(tickers))

def get_phase3_cns_sponsors():
    # Kept for future use — currently bypassed
    return set()

# ---------------- HYBRID FILTER (RELAXED 2025 VERSION) ----------------
def hybrid_filter(keyword_tickers, phase3_sponsors):
    print("Applying relaxed hybrid filter...")
    candidates = []
    for t in tqdm(keyword_tickers, desc="Market filtering"):
        try:
            time.sleep(0.05)
            info = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPreviousClose") or 0
            volume = info.get("averageVolume") or 0
            market_cap = info.get("marketCap") or 0

            if (0.15 <= price <= 35 and
                volume >= 30_000 and
                3_000_000 <= market_cap <= 2_000_000_000 and
                t.replace("^", "").isalnum()):
                candidates.append(t)
        except:
            continue

    new_candidates = [t for t in candidates if t not in KNOWN_MOONSHOTS]
    final_with_known = sorted(set(new_candidates + list(KNOWN_MOONSHOTS)))
    print(f"New candidates: {len(new_candidates)} | Total (with training): {len(final_with_known)}")
    return final_with_known

# ---------------- SCORING ----------------
def score_stock(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
        if hist.empty or len(hist) < 15:
            return None

        price = hist["Close"].iloc[-1]
        avg_vol_30d = hist["Volume"].iloc[-30:].mean() if len(hist) >= 30 else hist["Volume"].mean()
        current_vol = hist["Volume"].iloc[-1]

        if not (0.15 <= price <= 35) or avg_vol_30d < 35_000:
            return None

        low = hist["Close"].min()
        pct_from_low = (price - low) / low * 100 if low > 0 else 0
        vol_trend = current_vol / max(avg_vol_30d, 1)

        return {
            "ticker": ticker,
            "price": round(price, 3),
            "vol": int(avg_vol_30d),
            "% from low": round(pct_from_low, 1),
            "vol trend": round(vol_trend, 2)
        }
    except:
        return None

# ---------------- FAST BATCHED FEATURES ----------------
def add_features_batch(tickers, batch_size=60, sleep_between_batches=2.5):
    print(f"Fetching features for {len(tickers)} tickers in batches...")
    all_features = {}
    for i in tqdm(range(0, len(tickers), batch_size), desc="Feature batches"):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.Tickers(' '.join(batch))
            for t in batch:
                try:
                    hist = data.tickers[t].history(period="1y", auto_adjust=True)
                    info = data.tickers[t].info
                    if len(hist) < 60:
                        all_features[t] = None
                        continue
                    close = hist["Close"]
                    vol = hist["Volume"]
                    pct_from_low = (close.iloc[-1] / close.min()) - 1
                    days_since_low = (hist.index[-1] - close.idxmin()).days
                    delta = close.diff()
                    up = delta.clip(lower=0)
                    down = -delta.clip(upper=0)
                    ma_up = up.rolling(14).mean()
                    ma_down = down.rolling(14).mean()
                    rs = ma_up / (ma_down.replace(0, 0.0001))
                    rsi = 100 - (100 / (1 + rs))
                    rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
                    float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 1
                    shares_out = info.get("sharesOutstanding") or 1
                    float_ratio = float_shares / max(shares_out, 1)
                    short_interest = info.get("shortPercentOfFloat") or 0.0
                    vol_spike = vol.iloc[-1] / vol.iloc[-20:].mean() if len(vol) >= 20 else 1.0

                    all_features[t] = {
                        "pct_from_low": round(pct_from_low, 3),
                        "days_since_low": int(days_since_low),
                        "rsi": round(rsi_val, 2),
                        "float_ratio": round(float_ratio, 3),
                        "short_interest": round(short_interest, 3),
                        "vol_spike": round(vol_spike, 2)
                    }
                except:
                    all_features[t] = None
        except:
            for t in batch:
                all_features[t] = None
        if i + batch_size < len(tickers):
            time.sleep(sleep_between_batches)
    return all_features

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
    print("Starting NRXP-CLONE SCANNER v2025 — Hunting Fresh Moonshots\n")

    model, model_name = select_model()
    if model is None:
        print("Using fallback Random Forest...")
        model = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42)
        model_name = "Random Forest"

    keyword_tickers = fetch_keyword_biotechs()
    tickers = hybrid_filter(keyword_tickers, set())

    print("Scoring stocks...")
    scored = [score_stock(t) for t in tqdm(tickers) if score_stock(t)]
    scored = [s for s in scored if s is not None]
    if len(scored) < 20:
        print("Too few passed scoring.")
        return

    df = pd.DataFrame(scored)
    print(f"Scored & liquid: {len(df)} tickers")

    print("Adding technical features (batched)...")
    features_dict = add_features_batch(df["ticker"].tolist())
    df["features"] = df["ticker"].map(features_dict)
    df = df.dropna(subset=["features"]).reset_index(drop=True)
    if df.empty:
        print("No features extracted.")
        return

    feature_df = pd.DataFrame(df["features"].tolist())
    df = pd.concat([df.drop(columns=["features"]), feature_df], axis=1)

    print("Labeling...")
    df["label"] = df["ticker"].apply(label_stock)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    train_df = df[df["ticker"].isin(KNOWN_MOONSHOTS)].copy()
    predict_df = df[~df["ticker"].isin(KNOWN_MOONSHOTS)].copy()

    print(f"Training on {len(train_df)} known | Predicting on {len(predict_df)} new")

    cols = ["pct_from_low", "days_since_low", "rsi", "float_ratio", "short_interest", "vol_spike"]
    if len(train_df) >= 3:
        model.fit(train_df[cols], train_df["label"])
        probs = model.predict_proba(predict_df[cols])[:, 1]
    else:
        print("Not enough training data → using heuristic ranking")
        probs = (0.4 / (1 + predict_df["days_since_low"]/100) +
                 0.3 * predict_df["vol_spike"].clip(upper=10)/10 +
                 0.2 * (1 - predict_df["rsi"]/100) +
                 0.1 * (1 - predict_df["pct_from_low"].clip(upper=3)))

    predict_df = predict_df.copy()
    predict_df["predicted_prob"] = probs
    top_new = predict_df.sort_values("predicted_prob", ascending=False).head(20)

    top_new.to_csv(CSV_FILE, index=False)
    print(f"\nResults saved → {CSV_FILE}")
    print("\n" + "="*100)
    print(f" TOP 20 FRESH NRXP CLONES — {TODAY} | {model_name}")
    print("="*100)
    display_cols = ["ticker", "price", "vol", "% from low", "vol_spike", "rsi", "short_interest", "predicted_prob"]
    print(top_new[display_cols].round(4).to_string(index=False))
    print("="*100)

if __name__ == "__main__":
    main()
