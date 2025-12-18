import json
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tiktokpredict import TikTokTrueTrendModel 

# =========================
# Config
# =========================
DATA_PATH = "tiktok_data_10000.csv"
OUTPUT_HTML = "tiktok_trend_dashboard.html"

# === Manually blocked generic hashtags (avoid permanent dominance) ===
GENERIC_HASHTAGS = {
    "fyp", "foryou", "for you", "foryoupage",
    "viral", "xyzbca", "xyzabc", "fypシ", "fypシ゚",
    "fypage"
}

# Minimum training sample thresholds (too small -> empty results, avoid overfitting)
MIN_TRAIN_HASHTAG = 80
MIN_TRAIN_MUSIC = 40

# Music: filter "original sound"-type tracks (your original logic)
BLOCK_WORDS = [
    "original", "original sound", "originalton", "som original",
    "sonido original", "оригинальный", "звук", "orijinal", "ori"
]

# =========================
# Utils
# =========================
def _safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except:
        return 0

def _safe_float(x):
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except:
        return 0.0

def _is_fake_original_title(title: str) -> bool:
    t = str(title).lower()
    return any(w in t for w in BLOCK_WORDS)

def _clean_hashtag(tag: str) -> str:
    t = str(tag).strip().lower()
    t = t.lstrip("#")
    return t

# =========================
# Feature engineering (key: train vs score)
# =========================
def build_hashtag_features_from_weekly(weekly_ht: pd.DataFrame, require_next_week: bool) -> pd.DataFrame:
    """
    weekly_ht columns: week, hashtag, total_plays, video_count, engagement_rate
    require_next_week=True  -> build labeled training data (must have next_week)
    require_next_week=False -> build scoring data (does not require next_week, keep week=t)
    """
    df = weekly_ht.sort_values(["hashtag", "week"]).copy()

    for col in ["total_plays", "video_count", "engagement_rate"]:
        df[f"{col}_lag1"] = df.groupby("hashtag")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("hashtag")[col].shift(2)

    df["play_growth"] = (df["total_plays"] - df["total_plays_lag1"]) / (df["total_plays_lag1"] + 1)
    df["video_growth"] = (df["video_count"] - df["video_count_lag1"]) / (df["video_count_lag1"] + 1)
    df["engagement_growth"] = (df["engagement_rate"] - df["engagement_rate_lag1"]) / (df["engagement_rate_lag1"] + 1e-6)

    df["play_growth_lag1"] = df.groupby("hashtag")["play_growth"].shift(1)
    df["play_acceleration"] = df["play_growth"] - df["play_growth_lag1"]

    df["lifetime_plays"] = df.groupby("hashtag")["total_plays"].cumsum().shift(1).fillna(0)

    df["current_plays"] = df["total_plays"]
    df["current_video_count"] = df["video_count"]
    df["current_engagement"] = df["engagement_rate"]

    # Must have at least lag2
    df = df.dropna(subset=["total_plays_lag1", "total_plays_lag2"]).copy()

    # Filter small noise (tune if needed)
    df = df[df["lifetime_plays"] >= 500].copy()

    if require_next_week:
        df["next_week_plays"] = df.groupby("hashtag")["total_plays"].shift(-1)
        df["next_week_growth"] = (df["next_week_plays"] - df["total_plays"]) / (df["total_plays"] + 1)

        df = df.dropna(subset=["next_week_plays", "next_week_growth"]).copy()

        growth_threshold = df["next_week_growth"].quantile(0.75) if len(df) > 0 else 0.5
        df["is_trending_next_week"] = (
            (df["next_week_growth"] >= max(0.5, growth_threshold))
            & (df["next_week_plays"] >= 1000)
        ).astype(int)

    return df

def build_music_features_from_weekly(weekly_mu: pd.DataFrame, require_next_week: bool) -> pd.DataFrame:
    """
    weekly_mu columns: week, music_title_clean, total_plays, video_count, engagement_rate, is_original
    """
    df = weekly_mu.sort_values(["music_title_clean", "week"]).copy()

    for col in ["total_plays", "video_count", "engagement_rate"]:
        df[f"{col}_lag1"] = df.groupby("music_title_clean")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("music_title_clean")[col].shift(2)

    df["play_growth"] = (df["total_plays"] - df["total_plays_lag1"]) / (df["total_plays_lag1"] + 1)
    df["video_growth"] = (df["video_count"] - df["video_count_lag1"]) / (df["video_count_lag1"] + 1)
    df["engagement_growth"] = (df["engagement_rate"] - df["engagement_rate_lag1"]) / (df["engagement_rate_lag1"] + 1e-6)

    df["play_growth_lag1"] = df.groupby("music_title_clean")["play_growth"].shift(1)
    df["play_acceleration"] = df["play_growth"] - df["play_growth_lag1"]

    df["lifetime_plays"] = df.groupby("music_title_clean")["total_plays"].cumsum().shift(1).fillna(0)

    df["current_plays"] = df["total_plays"]
    df["current_video_count"] = df["video_count"]
    df["current_engagement"] = df["engagement_rate"]
    df["is_original_audio"] = (df["is_original"].astype(str).str.lower() == "t").astype(int)

    df = df.dropna(subset=["total_plays_lag1", "total_plays_lag2"]).copy()
    df = df[df["lifetime_plays"] >= 500].copy()

    if require_next_week:
        df["next_week_plays"] = df.groupby("music_title_clean")["total_plays"].shift(-1)
        df["next_week_growth"] = (df["next_week_plays"] - df["total_plays"]) / (df["total_plays"] + 1)

        df = df.dropna(subset=["next_week_plays", "next_week_growth"]).copy()

        growth_threshold = df["next_week_growth"].quantile(0.75) if len(df) > 0 else 0.5
        df["is_trending_next_week"] = (
            (df["next_week_growth"] >= max(0.5, growth_threshold))
            & (df["next_week_plays"] >= 1000)
        ).astype(int)

    return df

# =========================
# Train + score (dashboard only)
# =========================
HASHTAG_FEATURE_COLS = [
    "total_plays_lag1", "total_plays_lag2",
    "video_count_lag1", "video_count_lag2",
    "engagement_rate_lag1", "engagement_rate_lag2",
    "play_growth", "video_growth", "engagement_growth",
    "play_acceleration",
    "current_plays", "current_video_count", "current_engagement",
    "lifetime_plays",
]

MUSIC_FEATURE_COLS = [
    "total_plays_lag1", "total_plays_lag2",
    "video_count_lag1", "video_count_lag2",
    "engagement_rate_lag1", "engagement_rate_lag2",
    "play_growth", "video_growth", "engagement_growth",
    "play_acceleration",
    "current_plays", "current_video_count", "current_engagement",
    "lifetime_plays",
    "is_original_audio",
]

def fit_lr_and_score(train_df: pd.DataFrame, score_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Return trending probabilities for score_df.
    Training data must include is_trending_next_week.
    """
    if len(train_df) == 0 or len(score_df) == 0:
        return np.array([])

    # y must contain both 0 and 1, otherwise LR cannot be trained
    y = train_df["is_trending_next_week"].astype(int)
    if y.nunique() < 2:
        return np.array([])

    X_train = train_df[feature_cols].fillna(0.0)
    X_score = score_df[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_score_s = scaler.transform(X_score)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y)

    prob = clf.predict_proba(X_score_s)[:, 1]
    return prob

# =========================
# Precompute: for each week t -> predict ranking for t+1
# =========================
def build_weekly_predictions_for_dashboard(model: TikTokTrueTrendModel,
                                          top_k_hashtag=15,
                                          top_k_music=15) -> dict:
    max_week = int(model.df["week"].max())
    results = {}

    # Build full aggregates once (same as your gpt.py logic)
    model.build_weekly_hashtag_agg()
    model.build_weekly_music_agg()

    for t in range(2, max_week):  # use week=t to predict t+1
        ht_out, mu_out = [], []

        # -------------------------
        # Hashtag
        # -------------------------
        weekly_ht_t = model.weekly_hashtag[model.weekly_hashtag["week"] <= t].copy()
        if len(weekly_ht_t) > 0:
            # train: requires next_week (so valid rows up to t-1)
            ht_train = build_hashtag_features_from_weekly(weekly_ht_t, require_next_week=True)
            ht_train = ht_train[ht_train["week"] <= (t - 1)].copy()

            # score: does not require next_week, keep week=t
            ht_score_all = build_hashtag_features_from_weekly(weekly_ht_t, require_next_week=False)
            ht_score = ht_score_all[ht_score_all["week"] == t].copy()

            # Pre-score filtering: generic + small noise
            if len(ht_score) > 0:
                ht_score["hashtag_clean"] = ht_score["hashtag"].apply(_clean_hashtag)
                ht_score = ht_score[~ht_score["hashtag_clean"].isin(GENERIC_HASHTAGS)].copy()
                ht_score = ht_score[ht_score["current_plays"] >= 500].copy()

            if len(ht_train) >= MIN_TRAIN_HASHTAG and len(ht_score) > 0:
                prob = fit_lr_and_score(ht_train, ht_score, HASHTAG_FEATURE_COLS)

                if len(prob) == len(ht_score):
                    ht_score = ht_score.copy()
                    ht_score["trend_prob_raw"] = prob

                    penalty = np.log10(ht_score["lifetime_plays"] + 10)
                    ht_score["trend_score"] = ht_score["trend_prob_raw"] / (1.0 + penalty)

                    ht_score = ht_score.sort_values("trend_score", ascending=False).head(top_k_hashtag)

                    for _, r in ht_score.iterrows():
                        ht_out.append({
                            "hashtag": "#" + str(r["hashtag_clean"]),
                            "trend_prob": _safe_float(r.get("trend_prob_raw", 0.0)),
                            "trend_score": _safe_float(r.get("trend_score", 0.0)),
                            "lifetime_plays": _safe_int(r.get("lifetime_plays", 0)),
                            "thisweek_plays": _safe_int(r.get("current_plays", 0)),
                            "play_growth": _safe_float(r.get("play_growth", 0.0)),
                        })

        # -------------------------
        # Music
        # -------------------------
        weekly_mu_t = model.weekly_music[model.weekly_music["week"] <= t].copy()
        if len(weekly_mu_t) > 0:
            mu_train = build_music_features_from_weekly(weekly_mu_t, require_next_week=True)
            mu_train = mu_train[mu_train["week"] <= (t - 1)].copy()

            mu_score_all = build_music_features_from_weekly(weekly_mu_t, require_next_week=False)
            mu_score = mu_score_all[mu_score_all["week"] == t].copy()

            # Filter "original sound" type tracks
            if len(mu_score) > 0:
                mu_score = mu_score[~mu_score["music_title_clean"].apply(_is_fake_original_title)].copy()

            if len(mu_train) >= MIN_TRAIN_MUSIC and len(mu_score) > 0:
                prob = fit_lr_and_score(mu_train, mu_score, MUSIC_FEATURE_COLS)

                if len(prob) == len(mu_score):
                    mu_score = mu_score.copy()
                    mu_score["trend_prob_raw"] = prob

                    penalty = np.log10(mu_score["lifetime_plays"] + 10)
                    mu_score["trend_score"] = mu_score["trend_prob_raw"] / (1.0 + 0.5 * penalty)

                    mu_score = mu_score.sort_values("trend_score", ascending=False).head(top_k_music)

                    for _, r in mu_score.iterrows():
                        mu_out.append({
                            "music": str(r.get("music_title_clean", "")).strip(),
                            "trend_prob": _safe_float(r.get("trend_prob_raw", 0.0)),
                            "trend_score": _safe_float(r.get("trend_score", 0.0)),
                            "lifetime_plays": _safe_int(r.get("lifetime_plays", 0)),
                            "thisweek_plays": _safe_int(r.get("current_plays", 0)),
                            "play_growth": _safe_float(r.get("play_growth", 0.0)),
                        })

        results[str(t)] = {
            "week_t": t,
            "predict_week": t + 1,
            "hashtags": ht_out,
            "music": mu_out
        }

    return results

# =========================
# Single-file HTML (offline)
# =========================
def write_single_file_html(results, out_html=OUTPUT_HTML):
    weeks = sorted(results.keys(), key=lambda x: int(x))
    default_week = weeks[-1] if weeks else "0"
    data_json = json.dumps(results, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>TikTok Next-Week Trend Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }}
    .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-top: 16px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .muted {{ color: #666; }}
    select {{ padding: 8px 10px; border-radius: 8px; border: 1px solid #ccc; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
    th {{ text-align: left; background: #fafafa; position: sticky; top: 0; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f4f6; }}
    .hint {{ font-size: 12px; color: #777; margin-top: 6px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>TikTok Next-Week Trend Dashboard (Offline HTML)</h1>
  <div class="row">
    <div>
      <label class="muted">Select Week t (use data up to week t): </label>
      <select id="weekSelect"></select>
      <div class="hint">
        Week 0 starts at the earliest date in your dataset; each +7 days increments the week index.
      </div>
    </div>
    <div class="muted" id="weekInfo"></div>
  </div>

  <div class="card">
    <h2>Predicted Hot Hashtags for Week t+1</h2>
    <div class="hint">
      If "No results": often means week t had too few hashtags with ≥2 weeks history, or training set too small.
      Try lowering MIN_TRAIN_HASHTAG / filtering thresholds.
    </div>
    <div id="hashtagTable"></div>
  </div>

  <div class="card">
    <h2>Predicted Hot Music for Week t+1</h2>
    <div class="hint">
      Music is sparser in small datasets, so empty weeks happen more often.
      Try lowering MIN_TRAIN_MUSIC or loosening lifetime_plays threshold.
    </div>
    <div id="musicTable"></div>
  </div>

<script>
const DATA = {data_json};

function pct(x) {{
  return (100 * x).toFixed(1) + "%";
}}
function num(x) {{
  try {{ return Number(x).toLocaleString(); }} catch(e) {{ return String(x); }}
}}
function f2(x) {{
  const v = Number(x);
  if (Number.isNaN(v)) return "0.00";
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}}

function renderTable(rows, kind) {{
  if (!rows || rows.length === 0) {{
    return `<div class="muted">No results for this week.</div>`;
  }}

  const header = `
    <table>
      <thead>
        <tr>
          <th style="width:48px;">Rank</th>
          <th>${{kind==="hashtag" ? "Hashtag" : "Music"}}</th>
          <th>Trend Prob</th>
          <th>Trend Score</th>
          <th>Lifetime Plays</th>
          <th>This Week Plays</th>
          <th>Play Growth</th>
        </tr>
      </thead><tbody>`;

  const body = rows.map((r, i) => `
    <tr>
      <td><span class="pill">${{i+1}}</span></td>
      <td>${{kind==="hashtag" ? r.hashtag : r.music}}</td>
      <td>${{pct(r.trend_prob)}}</td>
      <td>${{Number(r.trend_score).toFixed(4)}}</td>
      <td>${{num(r.lifetime_plays)}}</td>
      <td>${{num(r.thisweek_plays)}}</td>
      <td>${{f2(r.play_growth)}}</td>
    </tr>
  `).join("");

  return header + body + "</tbody></table>";
}}

function setWeek(w) {{
  const obj = DATA[w];
  if (!obj) return;

  document.getElementById("weekInfo").innerText =
    `Week t = ${{obj.week_t}}  →  Predict Week t+1 = ${{obj.predict_week}}`;

  document.getElementById("hashtagTable").innerHTML = renderTable(obj.hashtags, "hashtag");
  document.getElementById("musicTable").innerHTML = renderTable(obj.music, "music");
}}

const select = document.getElementById("weekSelect");
const weeks = Object.keys(DATA).sort((a,b)=>Number(a)-Number(b));
weeks.forEach(w => {{
  const opt = document.createElement("option");
  opt.value = w;
  opt.textContent = `Week ${{w}}`;
  select.appendChild(opt);
}});
select.value = "{default_week}";
select.addEventListener("change", (e) => setWeek(e.target.value));
setWeek(select.value);
</script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Wrote dashboard HTML: {out_html}")
    print("   Double-click to open in browser. Works offline (single-file).")

# =========================
# main
# =========================
def main_generate_html():
    model = TikTokTrueTrendModel(DATA_PATH)
    model.load_data()

    results = build_weekly_predictions_for_dashboard(
        model,
        top_k_hashtag=15,
        top_k_music=15
    )

    write_single_file_html(results, out_html=OUTPUT_HTML)

if __name__ == "__main__":
    main_generate_html()
