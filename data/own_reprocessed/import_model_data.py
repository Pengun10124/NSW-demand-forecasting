from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parent
FEATURES = json.loads((ROOT / "feature_columns.json").read_text())
feat = pd.read_csv(ROOT / "daily_model_ready.csv", parse_dates=["Date"]).set_index("Date").sort_index()
assert feat.index.is_unique
assert feat[FEATURES + ["day_min", "mid_min"]].notna().all().all()
train = feat.loc[feat.index < "2026-04-01"]
hold = feat.loc["2026-04-01":"2026-06-30"]
july_august = feat.loc["2026-07-01":"2026-08-31"]
X_train = train[FEATURES]
X_hold = hold[FEATURES]
X_july_august = july_august[FEATURES]
y_train = train[["day_min", "mid_min"]]
y_hold = hold[["day_min", "mid_min"]]
y_july_august = july_august[["day_min", "mid_min"]]
print({"train_days": len(train), "holdout_days": len(hold), "july_august_days": len(july_august), "features": len(FEATURES)})
