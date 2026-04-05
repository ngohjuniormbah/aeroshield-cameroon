from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from core.feature_engineering import (
    add_time_and_lag_features,
    build_virtual_sensor_target,
    get_model_features,
)
from core.xlsx_loader import load_hackathon_xlsx


def train_model(xlsx_path: str | Path, outdir: str | Path) -> dict[str, float | str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df = load_hackathon_xlsx(xlsx_path)
    work = build_virtual_sensor_target(raw_df)
    model_df = add_time_and_lag_features(work)

    features = get_model_features()
    categorical = ["city", "region"]
    numeric = [col for col in features if col not in categorical]

    train_mask = model_df["time"] < pd.Timestamp("2025-01-01")
    train_df = model_df.loc[train_mask].copy()
    test_df = model_df.loc[~train_mask].copy()

    X_train = train_df[features]
    y_train = train_df["target_next_day_aqri"]
    X_test = test_df[features]
    y_test = test_df["target_next_day_aqri"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(features)),
        "project_name": "AeroShield Cameroon",
        "target_name": "target_next_day_aqri",
        "target_description": "Next-day Air Quality Risk Index forecast from a virtual sensor layer built on weather data.",
    }

    bundle = {
        "model": pipeline,
        "features": features,
        "metrics": metrics,
        "latest_data": model_df,
        "raw_data": raw_df,
    }

    joblib.dump(bundle, outdir / "aeroshield_model.joblib")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model_df.tail(500).to_csv(outdir / "demo_scoring_data.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AeroShield Cameroon model")
    parser.add_argument("--xlsx", required=True,
                        help="Path to the hackathon Excel dataset")
    parser.add_argument("--outdir", default="artifacts", help="Output folder")
    args = parser.parse_args()

    metrics = train_model(args.xlsx, args.outdir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
