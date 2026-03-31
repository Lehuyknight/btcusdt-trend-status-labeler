from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trend_core import BacktestParams, run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trend-detection CSV and print yearly net_pnl.")
    parser.add_argument("--input_csv", required=True, help="CSV with open_time, OHLCV, status")
    parser.add_argument("--fee_rate", type=float, default=0.0004)
    parser.add_argument("--slippage_rate", type=float, default=0.0002)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--chart_file", default="judgement_plot.png")
    return parser.parse_args()


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["open_time", "open", "high", "low", "close", "volume", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).sort_values("open_time").reset_index(drop=True)
    return df


def plot_wrong_regions(bt_df: pd.DataFrame, output_path: str) -> None:
    x = pd.to_datetime(bt_df["open_time"], unit="ms", utc=True)
    close = bt_df["close"]
    pos = bt_df["pos"]
    bar_pnl = bt_df["strategy_ret"]
    wrong = (pos != 0) & (bar_pnl < 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, close, lw=1.0, label="Close")
    ax.scatter(x[wrong], close[wrong], s=8, c="red", alpha=0.5, label="Wrong trend bars")
    ax.set_title("Trend detection errors (positioned but losing bar)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_input(args.input_csv)
    bt_params = BacktestParams(
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
    )
    bt_df, stats, yearly = run_backtest(df, bt_params)

    print("=== Overall ===")
    print(f"net_pnl: {stats['net_pnl'] * 100:.2f}%")
    print(f"max_drawdown: {stats['max_drawdown'] * 100:.2f}%")
    print(f"turnover: {stats['turnover']:.2f}")
    print(f"winrate: {stats['winrate'] * 100:.2f}%")

    print("\n=== Yearly net_pnl ===")
    for year, pnl in yearly.items():
        mark = "OK" if pnl > 0.20 else "LOW"
        print(f"{year}: {pnl * 100:.2f}% ({mark})")

    chart_file = Path(args.chart_file).as_posix()
    plot_wrong_regions(bt_df, chart_file)
    print(f"\nSaved chart: {chart_file}")


if __name__ == "__main__":
    main()
