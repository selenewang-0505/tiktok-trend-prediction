"""
IEOR 242A Fall 2025 - TikTok Time-Series Trend Prediction
True Trend Model (Avoid FYP/Viral Dominating)

Predicts:
1. Next-week trending hashtags
2. Next-week trending music

Key ideas:
- Aggregate by week
- Use growth / acceleration, NOT absolute volume
- Define "trending next week" based on growth of next week
- Penalize life-time huge tags (e.g., #fyp)
- Hard-filter generic hashtags (#fyp, #viral, #foryou, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore")

# === datapath ===
DATA_PATH = "tiktok_data_10000.csv"

# === mute some tags ===
GENERIC_HASHTAGS = {
    "fyp", "foryou", "for you", "foryoupage",
    "viral", "xyzbca", "xyzabc", "fypシ", "fypシ゚",
    "fypage"
}

class TikTokTrueTrendModel:
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.df = None

        # Weekly aggregates
        self.weekly_hashtag = None
        self.weekly_music = None

        # Feature-level data
        self.hashtag_features = None
        self.music_features = None

        # Fitted models
        self.hashtag_models = {}
        self.music_models = {}

    # =========================================================
    # 0. Load data
    # =========================================================
    def load_data(self):
        print("=" * 70)
        print("Loading TikTok data...")
        print("=" * 70)

        self.df = pd.read_csv(self.data_path)

        # datetime series
        self.df["event_time"] = pd.to_datetime(
            self.df["collected_time"], utc=True, errors="coerce"
        )
        self.df = self.df.dropna(subset=["event_time"]).copy()
        self.df["event_time"] = self.df["event_time"].dt.tz_convert(None)

        # only keep needed columns
        needed_cols = [
            "event_time",
            "challenges",
            "play_count",
            "digg_count",
            "comment_count",
            "share_count",
            "music_title",
            "music_original",
        ]
        self.df = self.df[needed_cols].copy()

        # remove rows with missing play_count
        self.df = self.df.dropna(subset=["play_count"]).copy()

        print(f"✓ Loaded {len(self.df)} rows")
        print(f"✓ Time range: {self.df['event_time'].min()} → {self.df['event_time'].max()}")

        # define week index
        base_date = self.df["event_time"].dt.normalize().min()
        self.df["week"] = ((self.df["event_time"].dt.normalize() - base_date)
                           .dt.days // 7).astype(int)

        print(f"✓ Data spans {self.df['week'].min()} → {self.df['week'].max()} weeks")

        # hashtags
        self.df["hashtags"] = self.df["challenges"].apply(self._extract_hashtags)

        return self.df

    def _extract_hashtags(self, s):
        """from 'challenges' column, extract list of hashtags"""
        if pd.isna(s) or s == "":
            return []
        # Usually a JSON list: ["tag1","tag2"]
        try:
            tags = json.loads(s)
            # Ensure it is a list
            if not isinstance(tags, list):
                return []
            return [str(t).strip().lower() for t in tags if t]
        except Exception:
            # Fallback: handle comma-separated strings, etc.
            try:
                parts = str(s).split(",")
                return [p.strip().lower() for p in parts if p.strip()]
            except Exception:
                return []

    # =========================================================
    # 1. Hashtag: Weekly aggregation
    # =========================================================
    def build_weekly_hashtag_agg(self):
        print("\n" + "=" * 70)
        print("Aggregating hashtags by week...")
        print("=" * 70)

        records = []
        for _, row in self.df.iterrows():
            week = row["week"]
            for tag in row["hashtags"]:
                records.append(
                    {
                        "week": week,
                        "hashtag": tag,
                        "play_count": row["play_count"],
                        "digg_count": row["digg_count"],
                        "comment_count": row["comment_count"],
                        "share_count": row["share_count"],
                    }
                )
        if not records:
            print("⚠ No hashtags found.")
            return None

        ht_df = pd.DataFrame(records)

        weekly = (
            ht_df.groupby(["week", "hashtag"])
            .agg(
                total_plays=("play_count", "sum"),
                video_count=("play_count", "count"),
                total_likes=("digg_count", "sum"),
                total_comments=("comment_count", "sum"),
                total_shares=("share_count", "sum"),
            )
            .reset_index()
        )

        weekly["engagement_rate"] = (
            weekly["total_likes"] + weekly["total_comments"] + weekly["total_shares"]
        ) / (weekly["total_plays"] + 1)

        self.weekly_hashtag = weekly
        print(f"✓ Weekly hashtag rows: {len(weekly)}")
        print(f"✓ Unique hashtags: {weekly['hashtag'].nunique()}")
        print(f"✓ Weeks: {weekly['week'].min()} → {weekly['week'].max()}")
        return weekly

    # =========================================================
    # 2. Hashtag: Time-series features (true trend)
    # =========================================================
    def build_hashtag_features(self):
        print("\n" + "=" * 70)
        print("Building hashtag time-series features...")
        print("=" * 70)

        df = self.weekly_hashtag.sort_values(["hashtag", "week"]).copy()

        # Lag features
        for col in ["total_plays", "video_count", "engagement_rate"]:
            df[f"{col}_lag1"] = df.groupby("hashtag")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("hashtag")[col].shift(2)

        # Growth
        df["play_growth"] = (df["total_plays"] - df["total_plays_lag1"]) / (
            df["total_plays_lag1"] + 1
        )
        df["video_growth"] = (df["video_count"] - df["video_count_lag1"]) / (
            df["video_count_lag1"] + 1
        )
        df["engagement_growth"] = (
            df["engagement_rate"] - df["engagement_rate_lag1"]
        ) / (df["engagement_rate_lag1"] + 1e-6)

        # Growth lag (for acceleration)
        df["play_growth_lag1"] = df.groupby("hashtag")["play_growth"].shift(1)
        df["play_acceleration"] = df["play_growth"] - df["play_growth_lag1"]

        # Lifetime plays (used to penalize extremely large but stable hashtags)
        df["lifetime_plays"] = (
            df.groupby("hashtag")["total_plays"].cumsum().shift(1).fillna(0)
        )

        # Current-week features
        df["current_plays"] = df["total_plays"]
        df["current_video_count"] = df["video_count"]
        df["current_engagement"] = df["engagement_rate"]

        # Define "next week trend"
        df["next_week_plays"] = df.groupby("hashtag")["total_plays"].shift(-1)

        # Next-week growth rate (relative to current week)
        df["next_week_growth"] = (
            df["next_week_plays"] - df["total_plays"]
        ) / (df["total_plays"] + 1)

        # Keep only rows with lag & next_week available
        df = df.dropna(
            subset=[
                "total_plays_lag1",
                "total_plays_lag2",
                "next_week_plays",
                "next_week_growth",
            ]
        ).copy()

        # Filter very small/noisy hashtags: too few lifetime plays
        df = df[df["lifetime_plays"] >= 500].copy()

        # Define trending based on next-week growth (>= +50% and next-week plays not too small)
        growth_threshold = df["next_week_growth"].quantile(0.75)  # upper quartile
        df["is_trending_next_week"] = (
            (df["next_week_growth"] >= max(0.5, growth_threshold))
            & (df["next_week_plays"] >= 1000)
        ).astype(int)

        print(f"✓ Samples with features: {len(df)}")
        print(
            f"✓ Trending next week (positive class): "
            f"{df['is_trending_next_week'].sum()} ({df['is_trending_next_week'].mean()*100:.1f}%)"
        )

        self.hashtag_features = df
        return df

    # =========================================================
    # 3. Hashtag: Train models
    # =========================================================
    def train_hashtag_models(self):
        print("\n" + "=" * 70)
        print("Training hashtag trend models...")
        print("=" * 70)

        df = self.hashtag_features.copy()

        # Time-based split
        max_week = df["week"].max()
        split_week = int(max_week * 0.7)

        train_df = df[df["week"] <= split_week]
        test_df = df[df["week"] > split_week]

        print(f"✓ Train weeks: <= {split_week}")
        print(f"✓ Test weeks:  > {split_week}")
        print(f"✓ Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        feature_cols = [
            "total_plays_lag1",
            "total_plays_lag2",
            "video_count_lag1",
            "video_count_lag2",
            "engagement_rate_lag1",
            "engagement_rate_lag2",
            "play_growth",
            "video_growth",
            "engagement_growth",
            "play_acceleration",
            "current_plays",
            "current_video_count",
            "current_engagement",
            "lifetime_plays",
        ]

        X_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df["is_trending_next_week"]
        X_test = test_df[feature_cols].fillna(0.0)
        y_test = test_df["is_trending_next_week"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                random_state=42,
            ),
        }

        for name, model in models.items():
            print("\n" + "-" * 50)
            print(f"Training {name}...")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = (
                model.predict_proba(X_test_s)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            self.hashtag_models[name] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "test_df": test_df,
            }

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 score:  {f1:.4f}")

        return self.hashtag_models

    # =========================================================
    # 4. Hashtag: Predict next week's trending hashtags
    # =========================================================
    def predict_next_week_hashtags(self, top_k=15):
        print("\n" + "=" * 70)
        print("Predicting NEXT WEEK trending hashtags (true trend model)...")
        print("=" * 70)

        # Select the model with the best F1 score
        best_name = max(self.hashtag_models, key=lambda k: self.hashtag_models[k]["f1"])
        best = self.hashtag_models[best_name]
        model = best["model"]
        scaler = best["scaler"]
        feature_cols = best["feature_cols"]

        print(f"✓ Using best model: {best_name} (F1 = {best['f1']:.3f})")

        df = self.hashtag_features.copy()
        latest_week = df["week"].max()
        latest_df = df[df["week"] == latest_week].copy()

        print(f"✓ Latest week index: {latest_week}")
        print(f"✓ Hashtags in latest week: {len(latest_df)}")

        X_latest = latest_df[feature_cols].fillna(0.0)
        X_latest_s = scaler.transform(X_latest)
        raw_prob = (
            model.predict_proba(X_latest_s)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_latest_s)
        )

        latest_df["trend_prob_raw"] = raw_prob

        # Apply penalty to huge lifetime-play hashtags to avoid #fyp always ranking first
        penalty = np.log10(latest_df["lifetime_plays"] + 10)
        latest_df["trend_score"] = latest_df["trend_prob_raw"] / (1.0 + penalty)

        # Filter out generic hashtags (hard blacklist)
        latest_df["hashtag_clean"] = latest_df["hashtag"].str.lstrip("#").str.lower()
        filtered = latest_df[
            ~latest_df["hashtag_clean"].isin(GENERIC_HASHTAGS)
        ].copy()

        # Remove very small/noisy tags (current-week plays too small)
        filtered = filtered[filtered["current_plays"] >= 500].copy()

        # Sort by trend_score
        filtered = filtered.sort_values("trend_score", ascending=False)

        top = filtered.head(top_k)

        print("\nRank  Hashtag                TrendProb   LifetimePlays   ThisWeekPlays   PlayGrowth")
        print("-" * 80)
        for i, (_, row) in enumerate(top.iterrows(), 1):
            tag = "#" + row["hashtag_clean"]
            print(
                f"{i:<4}  {tag:<22} "
                f"{row['trend_prob_raw']*100:6.1f}%   "
                f"{int(row['lifetime_plays']):>10}   "
                f"{int(row['current_plays']):>13}   "
                f"{row['play_growth']:+6.2f}"
            )

        return top

    # =========================================================
    # 5. Music: Weekly aggregation & features (same structure)
    # =========================================================
    def build_weekly_music_agg(self):
        print("\n" + "=" * 70)
        print("Aggregating music by week...")
        print("=" * 70)

        music_df = self.df.dropna(subset=["music_title"]).copy()
        music_df["music_title_clean"] = music_df["music_title"].astype(str).str.strip()

        weekly = (
            music_df.groupby(["week", "music_title_clean"])
            .agg(
                total_plays=("play_count", "sum"),
                video_count=("play_count", "count"),
                total_likes=("digg_count", "sum"),
                total_shares=("share_count", "sum"),
                is_original=("music_original", "first"),
            )
            .reset_index()
        )

        weekly["engagement_rate"] = (
            weekly["total_likes"] + weekly["total_shares"]
        ) / (weekly["total_plays"] + 1)

        self.weekly_music = weekly
        print(f"✓ Weekly music rows: {len(weekly)}")
        print(f"✓ Unique tracks: {weekly['music_title_clean'].nunique()}")
        print(f"✓ Weeks: {weekly['week'].min()} → {weekly['week'].max()}")
        return weekly

    def build_music_features(self):
        print("\n" + "=" * 70)
        print("Building music time-series features...")
        print("=" * 70)

        df = self.weekly_music.sort_values(["music_title_clean", "week"]).copy()

        for col in ["total_plays", "video_count", "engagement_rate"]:
            df[f"{col}_lag1"] = df.groupby("music_title_clean")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("music_title_clean")[col].shift(2)

        df["play_growth"] = (df["total_plays"] - df["total_plays_lag1"]) / (
            df["total_plays_lag1"] + 1
        )
        df["video_growth"] = (df["video_count"] - df["video_count_lag1"]) / (
            df["video_count_lag1"] + 1
        )
        df["engagement_growth"] = (
            df["engagement_rate"] - df["engagement_rate_lag1"]
        ) / (df["engagement_rate_lag1"] + 1e-6)

        df["play_growth_lag1"] = df.groupby("music_title_clean")["play_growth"].shift(1)
        df["play_acceleration"] = df["play_growth"] - df["play_growth_lag1"]

        df["lifetime_plays"] = (
            df.groupby("music_title_clean")["total_plays"].cumsum().shift(1).fillna(0)
        )

        df["current_plays"] = df["total_plays"]
        df["current_video_count"] = df["video_count"]
        df["current_engagement"] = df["engagement_rate"]
        df["is_original_audio"] = (df["is_original"] == "t").astype(int)

        df["next_week_plays"] = df.groupby("music_title_clean")["total_plays"].shift(-1)
        df["next_week_growth"] = (
            df["next_week_plays"] - df["total_plays"]
        ) / (df["total_plays"] + 1)

        df = df.dropna(
            subset=[
                "total_plays_lag1",
                "total_plays_lag2",
                "next_week_plays",
                "next_week_growth",
            ]
        ).copy()

        df = df[df["lifetime_plays"] >= 500].copy()

        growth_thr = df["next_week_growth"].quantile(0.75)
        df["is_trending_next_week"] = (
            (df["next_week_growth"] >= max(0.5, growth_thr))
            & (df["next_week_plays"] >= 1000)
        ).astype(int)

        print(f"✓ Music samples with features: {len(df)}")
        print(
            f"✓ Trending next week: {df['is_trending_next_week'].sum()} "
            f"({df['is_trending_next_week'].mean()*100:.1f}%)"
        )

        self.music_features = df
        return df

    def train_music_models(self):
        print("\n" + "=" * 70)
        print("Training music trend models...")
        print("=" * 70)

        df = self.music_features.copy()

        max_week = df["week"].max()
        split_week = int(max_week * 0.7)

        train_df = df[df["week"] <= split_week]
        test_df = df[df["week"] > split_week]

        print(f"✓ Train weeks: <= {split_week}")
        print(f"✓ Test weeks:  > {split_week}")
        print(f"✓ Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        feature_cols = [
            "total_plays_lag1",
            "total_plays_lag2",
            "video_count_lag1",
            "video_count_lag2",
            "engagement_rate_lag1",
            "engagement_rate_lag2",
            "play_growth",
            "video_growth",
            "engagement_growth",
            "play_acceleration",
            "current_plays",
            "current_video_count",
            "current_engagement",
            "lifetime_plays",
            "is_original_audio",
        ]

        X_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df["is_trending_next_week"]
        X_test = test_df[feature_cols].fillna(0.0)
        y_test = test_df["is_trending_next_week"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=3, random_state=42
            ),
        }

        for name, model in models.items():
            print("\n" + "-" * 50)
            print(f"Training {name}...")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = (
                model.predict_proba(X_test_s)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            self.music_models[name] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "test_df": test_df,
            }

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 score:  {f1:.4f}")

        return self.music_models

    def predict_next_week_music(self, top_k=15):
        print("\n" + "=" * 70)
        print("Predicting NEXT WEEK trending music...")
        print("=" * 70)

        best_name = max(self.music_models, key=lambda k: self.music_models[k]["f1"])
        best = self.music_models[best_name]
        model = best["model"]
        scaler = best["scaler"]
        feature_cols = best["feature_cols"]

        print(f"✓ Using best model: {best_name} (F1 = {best['f1']:.3f})")

        df = self.music_features.copy()
        latest_week = df["week"].max()
        latest_df = df[df["week"] == latest_week].copy()

        print(f"✓ Latest week index: {latest_week}")
        print(f"✓ Tracks in latest week: {len(latest_df)}")

        X_latest = latest_df[feature_cols].fillna(0.0)
        X_latest_s = scaler.transform(X_latest)
        raw_prob = (
            model.predict_proba(X_latest_s)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_latest_s)
        )

        latest_df["trend_prob_raw"] = raw_prob

        # Apply a mild penalty for huge but stable lifetime-play tracks
        penalty = np.log10(latest_df["lifetime_plays"] + 10)
        latest_df["trend_score"] = latest_df["trend_prob_raw"] / (1.0 + 0.5 * penalty)

        # ======================================================
        # REMOVE ALL "Original Sound" type fake tracks
        # ======================================================
        BLOCK_WORDS = [
            "original", "original sound", "originalton", "som original",
            "sonido original", "оригинальный", "звук", "orijinal", "ori"
        ]

        def is_fake_original_title(title):
            t = str(title).lower()
            return any(b in t for b in BLOCK_WORDS)

        latest_df = latest_df[~latest_df["music_title_clean"].apply(is_fake_original_title)].copy()

        # Sort
        latest_df = latest_df.sort_values("trend_score", ascending=False)

        top = latest_df.head(top_k)

        print("\nRank  Music Title                        TrendProb   LifetimePlays   ThisWeekPlays   PlayGrowth")
        print("-" * 100)
        for i, (_, row) in enumerate(top.iterrows(), 1):
            title = row["music_title_clean"]
            if len(title) > 30:
                title = title[:27] + "..."
            print(
                f"{i:<4}  {title:<32} "
                f"{row['trend_prob_raw']*100:6.1f}%   "
                f"{int(row['lifetime_plays']):>10}   "
                f"{int(row['current_plays']):>13}   "
                f"{row['play_growth']:+6.2f}"
            )

        return top


# =========================================================
# Main
# =========================================================

def main():
    model = TikTokTrueTrendModel(DATA_PATH)

    # Load
    model.load_data()

    # Hashtags
    model.build_weekly_hashtag_agg()
    model.build_hashtag_features()
    model.train_hashtag_models()
    top_hashtags = model.predict_next_week_hashtags(top_k=15)

    # Music
    model.build_weekly_music_agg()
    model.build_music_features()
    model.train_music_models()
    top_music = model.predict_next_week_music(top_k=15)

    return model, top_hashtags, top_music


if __name__ == "__main__":
    model, trending_hashtags, trending_music = main()
