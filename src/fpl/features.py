"""Feature engineering for FPL points prediction."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "merged_gw_2022_2026.parquet"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
ROLLING_COLS = ["minutes", "total_points", "expected_goals", "expected_assists", "bonus"]
TEAM_ROLLING_COLS = ["goals_conceded", "goals_scored"]
WINDOWS = [3, 5]
POS_COLS = ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]
TARGET = "total_points"
ID_COLS = ["season", "element", "name", "team", "opp_team", "GW", "kickoff_time", "xP"]


def load_raw_seasons(seasons=SEASONS, cache_path=RAW_PATH):
    """Load gameweek data for the given seasons (from cache if available), sorted in time order."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"Loaded cached data from {cache_path}")
    else:
        frames = []
        for season in seasons:
            url = DATA_URL.format(season=season)
            season_df = pd.read_csv(url)
            season_df["season"]=season
            frames.append(season_df)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cache_path, index=False)
        print("Downloaded and saved")

    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    df = df.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)

    return df



def add_rolling_features(df, cols, windows, group_cols=("season", "element")):
    """For each column and window, add the group's average over its previous rows."""
    df = df.copy()
    grouped = df.groupby(list(group_cols))
    for col in cols:
        for w in windows:
            new_name = f"{col}_avg_last_{w}"
            df[new_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
    return df


def build_team_matches(df, windows=WINDOWS):
    """One row per team per match, with goals conceded/scored and their rolling averages."""
    
    team_matches = (
        df.drop_duplicates(subset=["season", "fixture", "team"])
        [["season", "fixture", "team", "was_home", "kickoff_time", "team_h_score", "team_a_score"]]
        .copy()
    )

    team_matches = team_matches.sort_values(["season", "team", "kickoff_time"]).reset_index(drop=True)

    team_matches["goals_conceded"] = np.where(
        team_matches["was_home"],
        team_matches["team_a_score"],
        team_matches["team_h_score"]
    )

    team_matches["goals_scored"] = np.where(
        team_matches["was_home"],
        team_matches["team_h_score"],
        team_matches["team_a_score"]
    )

    team_matches = add_rolling_features(team_matches, TEAM_ROLLING_COLS, windows, group_cols=("season", "team"))

    return team_matches



def add_opponent_features(df, team_matches):
    """Attach each row's opponent's recent goals conceded/scored, plus the opponent's name."""
    opp_cols = [f"{c}_avg_last_{w}" for c in TEAM_ROLLING_COLS for w in WINDOWS]

    opp = team_matches[["season", "fixture", "was_home", "team"] + opp_cols].copy()
    opp["was_home"] = ~opp["was_home"]

    opp = opp.rename(columns={"team": "opp_team"}).rename(columns={c: f"opp_{c}" for c in opp_cols})

    df = df.merge(opp, on=["season", "fixture", "was_home"], how="left")

    return df


def add_context_features(df):
    """Drop manager rows, convert was_home to 0/1, and one-hot encode position."""
    
    df = df[df["position"] != "AM"].copy()
    df["was_home"] = df["was_home"].astype(int)
    position_dummies = pd.get_dummies(df["position"], prefix="pos", dtype=int)
    df = pd.concat([df, position_dummies], axis=1)
    df[["position", "pos_GK", "pos_DEF", "pos_MID", "pos_FWD", "was_home", "value"]]
    
    return df


def build_feature_table(seasons=SEASONS, save=True):
    """Run the full pipeline. Returns (model_df, feature_names)."""
    df = load_raw_seasons(seasons)
    df = add_rolling_features(df, ROLLING_COLS, WINDOWS)
    team_matches = build_team_matches(df)
    df = add_opponent_features(df, team_matches)
    df = add_context_features(df)

    avg_cols = [c for c in df.columns if "_avg_last_" in c]
    player_avg_cols = [c for c in avg_cols if not c.startswith("opp_")]
    df = df.dropna(subset=player_avg_cols)

    features = avg_cols + ["was_home", "value"] + POS_COLS
    model_df = df[ID_COLS + features + [TARGET]].copy()

    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_df.to_parquet(PROCESSED_PATH, index=False)

    return model_df, features
