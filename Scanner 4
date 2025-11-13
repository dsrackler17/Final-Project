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

# Random Forest
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
MIN_PRICE = 0.20
MAX_PRICE = 15
MIN_VOLUME = 50_000          # Relaxed
MIN_MARKET_CAP = 5_000_000   # Relaxed
MAX_MARKET_CAP = 1_000_000_000  # Relaxed

SOURCES = [
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nasdaq",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=nyse",
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=amex"
]

# Expanded & smarter keywords
KEYWORDS = [
    "bio", "pharma", "thera", "peutics", "labs", "tech", "gene", "cell", "neuro",
    "psych", "cns", "immune", "onco", "rare", "mental", "brain", "addict", "alzheimer",
    "parkinson", "schizo", "bipolar", "depress", "anxiety", "ptsd", "suicide"
]

TODAY = datetime.now().strftime("%b%d_%Y")
CSV_FILE = f"NRXP_clones_HYBRID_NEW_{TODAY}.csv"

# Known historical moonshots (for training only)
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
        prompt = f"\nEnter your choice (1-{max_choice}) [default: 1]: "
        choice = input(prompt).strip() or "1"
       
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

# ---------------- NAME NORMALIZATION ----------------
def normalize_name(name):
    if not name:
        return ""
    return name.upper() \
        .replace(" INC", "").replace(" INC.", "") \
        .replace(" CORPORATION", "").replace(" CORP", "") \
        .replace(" LTD", "").replace(" LIMITED", "") \
        .replace(" PHARMACEUTICALS", "").replace(" THERAPEUTICS", "") \
        .replace(" BIOPHARMA", "").replace(" BIOTECH", "") \
        .replace(",", "").replace(".", "").strip()

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
            exchange = url.split('=')[-1]
            print(f"Error scraping {exchange}: {e}")
        time.sleep(0.1)
    print(f"Biotech keyword tickers: {len(tickers)}")
    return sorted(list(tickers))

def get_phase3_cns_sponsors():
    base = "https://clinicaltrials.gov/api/v2/studies"
    expr = (
        'AREA[Phase]Phase 3 AND AREA[StudyType]Interventional AND AREA[InterventionType]Drug '
        'AND (AREA[ConditionSearch](depression OR bipolar OR suicide OR PTSD OR schizophrenia OR anxiety OR addiction OR Alzheimer OR Parkinson OR neurology OR psychiatry)) '
        'AND AREA[Status]NOT (Terminated OR Withdrawn OR Suspended)'
    )
    url = f"{base}?format=json&query.expr={expr}&pageSize=1000"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        data = resp.json()
    except Exception as e:
        print("ClinicalTrials.gov failed:", e)
        return set()
    
    sponsors = set()
    for study in data.get("studies", []):
        name = study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"]
        clean = normalize_name(name)
        if clean:
            sponsors.add(clean)
    print(f"Phase 3 CNS sponsors: {len(sponsors)}")
    return sponsors

# ---------------- HYBRID FILTER ----------------
def hybrid_filter(keyword_tickers, phase3_sponsors):
    print("Fetching company names for matching...")
    ticker_to_name = {}
    for t in tqdm(keyword_tickers, desc="Getting names"):
        try:
            info = yf.Ticker(t).info
            name = info.get("longName") or info.get("shortName") or ""
            if name:
                ticker_to_name[t] = name
        except:
            pass
        time.sleep(0.05)
    
    if not ticker_to_name:
        print("No company names fetched.")
        return []

    print("Normalizing names for fuzzy matching...")
    ticker_to_norm = {t: normalize_name(n) for t, n in ticker_to_name.items()}
    sponsors_norm = {normalize_name(s) for s in phase3_sponsors if s}

    matches = set()
    for ticker, norm_name in ticker_to_norm.items():
        for s_norm in sponsors_norm:
            if not s_norm or len(s_norm) < 3:
                continue
            # Match if sponsor substring or first 1-2 words
            words = s_norm.split()[:2]
            if s_norm in norm_name or any(w in norm_name for w in words if len(w) > 2):
                matches.add(ticker)
                break

    print(f"Hybrid matches (before known): {len(matches)}")

    # Apply market filters
    final = []
    for t in matches:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap") or 0
            price = info.get("currentPrice") or info.get("regularMarketPreviousClose") or 0
            vol = info.get("averageVolume") or 0
            if (MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP and
                MIN_PRICE <= price <= MAX_PRICE and
                vol >= MIN_VOLUME):
                final.append(t)
        except:
            pass
        time.sleep(0.05)

    # Add known moonshots for training (but mark them)
    final_with_known = list(set(final) | KNOWN_MOONSHOTS)
    print(f"Final candidates (incl. known for training): {len(final_with_known)}")
    print(f"New candidates (excl. known): {len(set(final) - KNOWN_MOONSHOTS)}")
    return sorted(final_with_known)

# ---------------- FEATURES & LABELS ----------------
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
        float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 1
        shares_out = info.get("sharesOutstanding") or 1
        float_ratio = float_shares / max(shares_out, 1)
        short_interest = info.get("shortPercentOfFloat") or 0
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
    print("Starting HYBRID NRXP-CLONE SCANNER (New Predictions Only)\n")
   
    model, model_name = select_model()
    if model is None:
        print("No model available. Install XGBoost or use Random Forest.")
        return
   
    keyword_tickers = fetch_keyword_biotechs()
    phase3_sponsors = get_phase3_cns_sponsors()
    tickers = hybrid_filter(keyword_tickers, phase3_sponsors)
   
    if len(tickers) < 3:
        print("Not enough tickers found.")
        return
   
    print("Scoring stocks...")
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

    print("Adding technical features...")
    df["features"] = df["ticker"].apply(add_features)
    df = df.dropna(subset=["features"]).copy()
    if df.empty:
        print("No ticker had enough data.")
        return

    feature_df = pd.DataFrame(df["features"].tolist(), index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    df.drop(columns=["features"], inplace=True)
   
    print("Labeling historical moonshots...")
    df["label"] = df["ticker"].apply(label_stock)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if len(df) < 5:
        print("Not enough labeled data.")
        return

    # Split: Train on known, predict on new
    train_df = df[df["ticker"].isin(KNOWN_MOONSHOTS)].copy()
    predict_df = df[~df["ticker"].isin(KNOWN_MOONSHOTS)].copy()

    print(f"Training on {len(train_df)} known moonshots")
    print(f"Predicting on {len(predict_df)} new candidates")

    if len(train_df) < 3 or len(predict_df) == 0:
        print("Not enough data for training/prediction split.")
        return

    cols = ["pct_from_low", "days_since_low", "rsi", "float_ratio", "short_interest", "vol_spike"]
    X_train = train_df[cols]
    y_train = train_df["label"]

    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    X_pred = predict_df[cols]
    predict_df["predicted_prob"] = model.predict_proba(X_pred)[:, 1]

    top_new = predict_df.sort_values("predicted_prob", ascending=False).head(20)
    
    if top_new.empty:
        print("No new high-probability clones found.")
        return

    top_new.to_csv(CSV_FILE, index=False)
    print(f"\nResults saved to: {CSV_FILE}")
    print("\n" + "="*90)
    print(f" TOP 20 *NEW* NRXP CLONES — {len(top_new)} FOUND ({TODAY}) | Model: {model_name}")
    print("="*90)
    display_cols = ["ticker", "price", "vol", "% from low", "vol_spike", "rsi", "short_interest", "predicted_prob"]
    print(top_new[display_cols].round(3).to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    main()
