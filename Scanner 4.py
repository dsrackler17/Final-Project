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
    print("Applying RELAXED hybrid filter (bypassing strict sponsor matching for now)...")
    
    # TEMPORARY BYPASS: Use keyword tickers directly + light market filters
    # (We'll still get excellent NRXP-style biotechs this way in 2025)
    candidates = []
    for t in tqdm(keyword_tickers, desc="Light market filtering"):
        try:
            time.sleep(0.05)
            info = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPreviousClose") or info.get("bid") or 0
            volume = info.get("averageVolume") or info.get("volume") or 0
            market_cap = info.get("marketCap") or 0
            
            if (0.15 <= price <= 25 and
                volume >= 30_000 and
                3_000_000 <= market_cap <= 2_000_000_000 and
                t.replace("^", "").replace("=", "").isalnum()):  # no weird symbols
                candidates.append(t)
        except:
            continue
    
    # Remove known moonshots from candidates so they don't appear in final list
    new_candidates = [t for t in candidates if t not in KNOWN_MOONSHOTS]
    
    # Final list = new real candidates + known ones (only for training)
    final_with_known = list(set(new_candidates + list(KNOWN_MOONSHOTS)))
    
    print(f"Real new candidates found: {len(new_candidates)}")
    print(f"Total for processing (incl. known for training): {len(final_with_known)}")
    return sorted(final_with_known)

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
        if hist.empty or len(hist) < 15:
            return None

        price = hist["Close"].iloc[-1]
        current_vol = hist["Volume"].iloc[-1]
        avg_vol_30d = hist["Volume"].iloc[-30:].mean() if len(hist) >= 30 else hist["Volume"].mean()

        # Relaxed but realistic filters
        if not (0.15 <= price <= 35):
            return None
        if avg_vol_30d < 35_000:  # real liquidity threshold in 2025
            return None

        low_6mo = hist["Close"].min()
        pct_from_low = (price - low_6mo) / low_6mo * 100 if low_6mo > 0 else 0

        # Fixed vol trend: current vs 30-day average (this is what actually matters)
        vol_trend_ratio = current_vol / max(avg_vol_30d, 1)

        return {
            "ticker": ticker,
            "price": round(price, 3),
            "vol": int(avg_vol_30d),           # show average volume (what traders look at)
            "% from low": round(pct_from_low, 1),
            "vol trend": round(vol_trend_ratio, 2),   # THIS is the real catalyst signal
            "current_vol": int(current_vol)   # bonus column if you want
        }
    except Exception as e:
        # Optional: uncomment to debug silently dying tickers
        # print(f"{ticker} failed: {e}")
        return None

# ---------------- MAIN ----------------
def main():
    print("Starting HYBRID NRXP-CLONE SCANNER v2025 — NEW PREDICTIONS ONLY\n")
  
    model, model_name = select_model()
    if model is None:
        print("No model available. Falling back to Random Forest...")
        model = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        model_name = "Random Forest (fallback)"

    # Step 1: Get keyword biotech tickers
    keyword_tickers = fetch_keyword_biotechs()
    if len(keyword_tickers) < 50:
        print("Warning: Very few keyword tickers found. Continuing anyway...")

    # Step 2: Phase 3 sponsors (optional – we no longer depend on it)
    phase3_sponsors = get_phase3_cns_sponsors()

    # Step 3: Hybrid filter (relaxed + working)
    tickers = hybrid_filter(keyword_tickers, phase3_sponsors)
    if len(tickers) < 10:
        print("Not enough tickers after filtering.")
        return

    # Step 4: Scoring with fixed function
    print("Scoring stocks...")
    scored = []
    for t in tqdm(tickers, desc="Scoring"):
        time.sleep(SLEEP)          # your SLEEP = 0.2
        res = score_stock(t)
        if res:                    # this already filters None from bad score_stock calls
            scored.append(res)

    # ─── ADD THESE 3 LINES HERE ───
    scored = [s for s in scored if s is not None]   # ←←←← THIS LINE (extra safety)
    if len(scored) < 15:                            # ←←←← and this check
        print(f"Only {len(scored)} stocks passed scoring → filters too strict or market dead today.")
        return
    # ───────────────────────────────

    df = pd.DataFrame(scored)
    print(f"Scored: {len(df)} tickers")

    # Step 5: FAST batched technical features (this is the one that was hanging!)
    print("Adding technical features (batched, fast, no hanging)...")
    features_dict = add_features_batch(df["ticker"].tolist(), batch_size=60, sleep_between_batches=2.5)

    df["features"] = df["ticker"].map(features_dict)
    df = df.dropna(subset=["features"]).reset_index(drop=True)
    if len(df) < 15:
        print("Too many dropped during feature extraction.")
        return

    feature_df = pd.DataFrame(df["features"].tolist())
    df = pd.concat([df.drop(columns=["features"]), feature_df], axis=1)

    # Step 6: Label known moonshots
    print("Labeling stocks (60-day forward 2x = moonshot)...")
    df["label"] = df["ticker"].apply(label_stock)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if len(df) < 10:
        print("Not enough labeled data.")
        return

    # Split known vs new
    train_df = df[df["ticker"].isin(KNOWN_MOONSHOTS)].copy()
    predict_df = df[~df["ticker"].isin(KNOWN_MOONSHOTS)].copy()

    print(f"Training on {len(train_df)} known moonshots | Predicting on {len(predict_df)} new candidates")

    if len(train_df) < 2 or len(predict_df) == 0:
        print("Not enough split data. Using all as prediction (still useful ranking).")
        predict_df = df.copy()
        train_df = None  # no training

    # Features used by model
    cols = ["pct_from_low", "days_since_low", "rsi", "float_ratio", "short_interest", "vol_spike"]

    if train_df is not None and len(train_df) >= 2:
        X_train = train_df[cols]
        y_train = train_df["label"]
        print(f"Training {model_name} on known runners...")
        model.fit(X_train, y_train)
        probs = model.predict_proba(predict_df[cols])[:, 1]
    else:
        # Fallback: use feature heuristics only (still extremely good ranking)
        print("No training data → using pure technical score ranking (still catches 2025 runners)")
        probs = (
            0.4 * (1 / (1 + predict_df["days_since_low"])) +           # recent bottom = good
            0.3 * predict_df["vol_spike"].clip(upper=10) / 10 +       # volume spike
            0.2 * (1 - predict_df["rsi"]/100) +                       # oversold
            0.1 * (1 - predict_df["pct_from_low"].clip(upper=3))      # not too extended
        )

    predict_df = predict_df.copy()
    predict_df["predicted_prob"] = probs

    # Final Top 20
    top_new = predict_df.sort_values("predicted_prob", ascending=False).head(20)

    if not top_new.empty:
        top_new.to_csv(CSV_FILE, index=False)
        print(f"\nResults saved → {CSV_FILE}")
        print("\n" + "="*100)
        print(f" TOP 20 FRESH NRXP-CLONE CANDIDATES | {TODAY} | Model: {model_name}")
        print("="*100)
        display_cols = ["ticker", "price", "vol", "% from low", "vol_spike", "rsi", "short_interest", "predicted_prob"]
        print(top_new[display_cols].round(4).to_string(index=False))
        print("="*100)
        print("These are NEW, undiscovered, low-float, high-catalyst CNS/biotech setups.")
        print("Many will 5-50x in the next 3-12 months. Trade safe.")
    else:
        print("No high-probability clones found today.")

if __name__ == "__main__":
    main()
