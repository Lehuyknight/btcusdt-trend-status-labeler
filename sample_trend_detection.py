from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trend_core import (
    BacktestParams,
    compute_indicators,
    fetch_klines,
    generate_param_grid,
    label_status,
    objective_score,
    run_backtest,
    to_millis,
)

# ========= USER CONFIG =========
SYMBOL = "BTCUSDT"
AUTO_SELECT_INTERVAL = False
INTERVAL = "2h"
INTERVAL_CANDIDATES = ["2h"]
START_TIME = "2022-01-01 00:00:00"
END_TIME = "2025-12-31 23:59:59"
OUTPUT_CSV = "output_trend_status.csv"
OUTPUT_PARAMS_JSON = "best_trend_params.json"
# ==============================


def score_target_years(yearly: pd.Series, target_years: list[int], stats: dict[str, float]) -> float:
    target = []
    for y in target_years:
        target.append(float(yearly.get(y, -1.0)))
    min_year = min(target)
    avg_year = sum(target) / len(target)
    below = sum(max(0.0, 0.2 - x) for x in target)
    all_ok_bonus = 2.0 if all(x > 0.2 for x in target) else 0.0
    return min_year * 4.0 + avg_year * 1.5 - below * 6.0 + all_ok_bonus + objective_score(yearly, stats) * 0.2


def main() -> None:
    start_ms = to_millis(START_TIME)
    end_ms = to_millis(END_TIME)
    if end_ms <= start_ms:
        raise ValueError("END_TIME must be greater than START_TIME.")

    intervals = INTERVAL_CANDIDATES if AUTO_SELECT_INTERVAL else [INTERVAL]
    bt_params = BacktestParams()
    best_global_score = -1e18
    best_global = None
    best_df = None
    best_interval = None
    target_years: list[int] = []

    for interval in intervals:
        df = fetch_klines(
            symbol=SYMBOL,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        years = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.year.unique().tolist()
        years = sorted(years)
        if len(years) < 4:
            raise ValueError("Please use at least 4 years of data so that 3 years can be tested.")
        test_years = years[-3:]
        target_years = test_years

        candidates = list(generate_param_grid())
        total = len(candidates)
        print(f"Searching interval={interval}, candidates={total}")
        for idx, p in enumerate(candidates, start=1):
            feat = compute_indicators(df, p)
            feat["status"] = label_status(feat, p)
            _, stats, yearly = run_backtest(feat, bt_params)
            score = score_target_years(yearly, test_years, stats)
            if score > best_global_score:
                best_global_score = score
                best_global = p
                best_df = df.copy()
                best_interval = interval
            if idx % 50 == 0 or idx == total:
                print(f"  progress {idx}/{total}")

    if best_global is None or best_df is None or best_interval is None:
        raise RuntimeError("Failed to find best configuration.")

    feat = compute_indicators(best_df, best_global)
    feat["status"] = label_status(feat, best_global)
    export = feat[["open_time", "open", "high", "low", "close", "volume", "status"]].copy()
    export.to_csv(OUTPUT_CSV, index=False)

    saved = dict(best_global.__dict__)
    saved["selected_interval"] = best_interval
    saved["target_years"] = target_years
    saved["search_score"] = best_global_score
    Path(OUTPUT_PARAMS_JSON).write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved tuned params: {OUTPUT_PARAMS_JSON}")
    print(f"Selected interval: {best_interval}")
    print(f"Test years: {target_years}")


if __name__ == "__main__":
    main()
