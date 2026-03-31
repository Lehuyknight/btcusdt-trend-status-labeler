from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"


@dataclass
class TrendParams:
    adx_period: int = 14
    atr_period: int = 14
    ema_fast: int = 21
    ema_slow: int = 55
    adx_enter: float = 24.0
    adx_exit: float = 18.0
    atr_pct_floor: float = 0.0015
    slope_floor: float = 0.0
    hold_bars: int = 3
    atr_stop_k: float = 2.0
    breakout_lookback: int = 20
    breakout_buffer: float = 0.0


@dataclass
class BacktestParams:
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    leverage: float = 1.0
    initial_equity: float = 1.0


def interval_to_milliseconds(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def to_millis(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    import requests

    out: List[list] = []
    cursor = start_ms
    step = interval_to_milliseconds(interval)
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        r = requests.get(BINANCE_FUTURES_KLINES, params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        last_open = int(batch[-1][0])
        cursor = last_open + step
        if len(batch) < 1500:
            break

    if not out:
        raise RuntimeError("No kline data fetched.")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(out, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame, p: TrendParams) -> pd.DataFrame:
    out = df.copy()
    h = out["high"]
    l = out["low"]
    c = out["close"]

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / p.atr_period, adjust=False, min_periods=p.atr_period).mean()

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=out.index)
    minus_dm = pd.Series(minus_dm, index=out.index)

    plus_di = 100 * plus_dm.ewm(alpha=1.0 / p.adx_period, adjust=False, min_periods=p.adx_period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / p.adx_period, adjust=False, min_periods=p.adx_period).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / p.adx_period, adjust=False, min_periods=p.adx_period).mean()

    ema_fast = c.ewm(span=p.ema_fast, adjust=False, min_periods=p.ema_fast).mean()
    ema_slow = c.ewm(span=p.ema_slow, adjust=False, min_periods=p.ema_slow).mean()
    ema_slow_slope = ema_slow.diff() / ema_slow.shift(1).replace(0, np.nan)

    out["atr"] = atr
    out["atr_pct"] = atr / c.replace(0, np.nan)
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = adx
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["ema_slow_slope"] = ema_slow_slope.fillna(0.0)
    out["atr"] = atr
    out["roll_high_prev"] = h.rolling(window=p.breakout_lookback, min_periods=p.breakout_lookback).max().shift(1)
    out["roll_low_prev"] = l.rolling(window=p.breakout_lookback, min_periods=p.breakout_lookback).min().shift(1)
    return out


def label_status(df: pd.DataFrame, p: TrendParams) -> pd.Series:
    status = np.zeros(len(df), dtype=np.int8)
    hold_count = 0
    prev = 0
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        adx = float(row["adx"]) if not np.isnan(row["adx"]) else 0.0
        atr_pct = float(row["atr_pct"]) if not np.isnan(row["atr_pct"]) else 0.0
        atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
        slope = float(row["ema_slow_slope"]) if not np.isnan(row["ema_slow_slope"]) else 0.0
        close = float(row["close"])
        ema_slow = float(row["ema_slow"]) if not np.isnan(row["ema_slow"]) else close
        plus_di = float(row["plus_di"]) if not np.isnan(row["plus_di"]) else 0.0
        minus_di = float(row["minus_di"]) if not np.isnan(row["minus_di"]) else 0.0
        roll_high_prev = float(row["roll_high_prev"]) if not np.isnan(row["roll_high_prev"]) else close
        roll_low_prev = float(row["roll_low_prev"]) if not np.isnan(row["roll_low_prev"]) else close

        trend_ok_enter = adx >= p.adx_enter and atr_pct >= p.atr_pct_floor
        trend_ok_hold = adx >= p.adx_exit and atr_pct >= p.atr_pct_floor * 0.8

        direction_hold = 0
        if close > ema_slow and plus_di >= minus_di and slope >= p.slope_floor:
            direction_hold = 1
        elif close < ema_slow and minus_di > plus_di and slope <= -p.slope_floor:
            direction_hold = -1

        direction_enter = 0
        long_break = close >= roll_high_prev * (1.0 + p.breakout_buffer)
        short_break = close <= roll_low_prev * (1.0 - p.breakout_buffer)
        if direction_hold == 1 and long_break:
            direction_enter = 1
        elif direction_hold == -1 and short_break:
            direction_enter = -1

        new_status = prev
        # ATR-based stop: nếu đang có vị thế và giá đi ngược > k * ATR thì flat ngay.
        if prev != 0 and entry_price > 0.0 and entry_atr > 0.0:
            if prev == 1 and close <= entry_price - p.atr_stop_k * entry_atr:
                new_status = 0
                status[i] = new_status
                prev = 0
                hold_count = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            if prev == -1 and close >= entry_price + p.atr_stop_k * entry_atr:
                new_status = 0
                status[i] = new_status
                prev = 0
                hold_count = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue

        if prev == 0:
            if trend_ok_enter and direction_enter != 0:
                new_status = direction_enter
                hold_count = 1
                entry_price = close
                entry_atr = atr_val
            else:
                new_status = 0
        else:
            if hold_count < p.hold_bars:
                hold_count += 1
                new_status = prev
            else:
                if not trend_ok_hold:
                    new_status = 0
                    hold_count = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                elif direction_hold == 0:
                    new_status = 0
                    hold_count = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                elif direction_hold != prev:
                    if trend_ok_enter:
                        new_status = direction_enter if direction_enter != 0 else 0
                        hold_count = 1
                        if new_status != 0:
                            entry_price = close
                            entry_atr = atr_val
                    else:
                        new_status = 0
                        hold_count = 0
                        entry_price = 0.0
                        entry_atr = 0.0
                else:
                    new_status = prev
                    hold_count += 1

        status[i] = new_status
        prev = int(new_status)

    return pd.Series(status, index=df.index, name="status")


def run_backtest(df: pd.DataFrame, p: BacktestParams) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    out = df.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["pos"] = out["status"].shift(1).fillna(0).astype(float)
    out["trade_change"] = (out["pos"] - out["pos"].shift(1).fillna(0)).abs()
    out["turnover"] = out["trade_change"]
    out["cost"] = out["turnover"] * (p.fee_rate + p.slippage_rate)
    out["strategy_ret"] = out["pos"] * out["ret"] * p.leverage - out["cost"]
    out["equity"] = p.initial_equity * (1.0 + out["strategy_ret"]).cumprod()
    out["cummax"] = out["equity"].cummax()
    out["dd"] = out["equity"] / out["cummax"] - 1.0
    out["year"] = pd.to_datetime(out["open_time"], unit="ms", utc=True).dt.year

    net = float(out["equity"].iloc[-1] / p.initial_equity - 1.0)
    mdd = float(out["dd"].min())
    turnover = float(out["turnover"].sum())
    winrate = float((out["strategy_ret"] > 0).mean())
    stats = {
        "net_pnl": net,
        "max_drawdown": mdd,
        "turnover": turnover,
        "winrate": winrate,
    }
    yearly = out.groupby("year")["strategy_ret"].apply(lambda x: (1.0 + x).prod() - 1.0)
    return out, stats, yearly


def objective_score(yearly: pd.Series, stats: Dict[str, float]) -> float:
    if yearly.empty:
        return -1e9
    min_year = float(yearly.min())
    avg_year = float(yearly.mean())
    dd_penalty = max(0.0, abs(stats["max_drawdown"]) - 0.30) * 2.0
    turn_penalty = max(0.0, stats["turnover"] - 800) * 0.0005
    return min_year * 2.5 + avg_year - dd_penalty - turn_penalty


def generate_param_grid() -> Iterable[TrendParams]:
    adx_enter_list = [24.0, 26.0]
    adx_exit_list = [18.0]
    atr_floor_list = [0.0012, 0.0018]
    ema_fast_list = [13]
    ema_slow_list = [55, 89]
    hold_list = [2, 4]
    slope_floor_list = [0.0, 0.0001]
    breakout_lookback_list = [20, 36]
    breakout_buffer_list = [0.0, 0.0005]

    for adx_enter in adx_enter_list:
        for adx_exit in adx_exit_list:
            if adx_exit >= adx_enter:
                continue
            for atr_floor in atr_floor_list:
                for ema_fast in ema_fast_list:
                    for ema_slow in ema_slow_list:
                        if ema_fast >= ema_slow:
                            continue
                        for hold in hold_list:
                            for slope_floor in slope_floor_list:
                                for breakout_lookback in breakout_lookback_list:
                                    for breakout_buffer in breakout_buffer_list:
                                        yield TrendParams(
                                            ema_fast=ema_fast,
                                            ema_slow=ema_slow,
                                            adx_enter=adx_enter,
                                            adx_exit=adx_exit,
                                            atr_pct_floor=atr_floor,
                                            hold_bars=hold,
                                            slope_floor=slope_floor,
                                            breakout_lookback=breakout_lookback,
                                            breakout_buffer=breakout_buffer,
                                        )


def walk_forward_split(df: pd.DataFrame, test_years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.year
    test_mask = years.isin(test_years)
    pre_test = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    if pre_test.empty:
        raise ValueError("Not enough history before test years for tuning.")
    split_idx = int(len(pre_test) * 0.8)
    train_df = pre_test.iloc[:split_idx].copy()
    val_df = pre_test.iloc[split_idx:].copy()
    return train_df, val_df, test_df


def tune_params(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    bt_params: BacktestParams,
) -> TrendParams:
    candidates = list(generate_param_grid())
    total = len(candidates)
    best = None
    best_score = -1e18
    for idx, p in enumerate(candidates, start=1):
        tr_feat = compute_indicators(train_df, p)
        tr_feat["status"] = label_status(tr_feat, p)
        _, tr_stats, tr_year = run_backtest(tr_feat, bt_params)
        tr_score = objective_score(tr_year, tr_stats)

        val_feat = compute_indicators(val_df, p)
        val_feat["status"] = label_status(val_feat, p)
        _, val_stats, val_year = run_backtest(val_feat, bt_params)
        val_score = objective_score(val_year, val_stats)

        score = tr_score * 0.3 + val_score * 0.7
        if score > best_score:
            best_score = score
            best = p
        if idx % 20 == 0 or idx == total:
            print(f"Tuning progress: {idx}/{total} candidates")

    if best is None:
        raise RuntimeError("Unable to tune parameters.")
    return best
