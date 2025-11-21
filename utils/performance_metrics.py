"""
Professional performance metrics calculator for trading strategies.

Features:
- Comprehensive risk-adjusted metrics (Sharpe, Sortino, CAGR)
- ENHANCED: Intraday Sharpe/Sortino with conservative annualization
- Robust drawdown calculation with timestamps
- Multi-directional position tracking (LONG/SHORT)
- Open P/L calculation with live price lookup
- Handles edge cases and missing data gracefully

Conservative mode (Option A):
- No annualization if equity timespan < 2 days
- Intraday annualization factor capped at sqrt(252)
- CAGR only computed if timespan >= 30 days
- Sortino clipped to 99.99 when downside deviation ~ 0
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Tuple, List, Any
import warnings


# ============================================================================
# CONSTANTS
# ============================================================================
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5  # US market: 9:30 AM - 4:00 PM ET
TRADING_MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60

MIN_DAILY_RETURNS_FOR_SHARPE = 10  # Minimum daily returns needed
MIN_RETURNS_FOR_SHARPE = 5        # Absolute minimum returns needed


# ============================================================================
# ROBUST DRAWDOWN CALCULATION
# ============================================================================
def max_drawdown_stats(
    pnl: pd.Series,
    timestamps: pd.Series | None = None,
    start_equity: float = 0.0
):
    """
    Calculate maximum drawdown with detailed statistics.
    
    Args:
        pnl: Series of P&L values
        timestamps: Optional timestamp series for duration calculation
        start_equity: Starting equity value
    
    Returns:
        Tuple of (mdd_abs, mdd_pct, dd_start, dd_end, dd_duration)
    """
    if pnl is None or len(pnl) == 0:
        return 0.0, 0.0, None, None, pd.Timedelta(0)

    equity = pnl.fillna(0).cumsum() + float(start_equity)
    peak = equity.cummax()

    dd_abs = peak - equity
    mdd_abs = float(dd_abs.max() if not dd_abs.empty else 0.0)

    if equity.empty:
        return 0.0, 0.0, None, None, pd.Timedelta(0)

    # Calculate percentage drawdown safely
    peak_np = peak.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_vals = np.where(peak_np == 0, np.nan, equity.to_numpy() / peak_np - 1.0)

    pct_series = pd.Series(pct_vals, index=equity.index)

    if len(pct_series) == 0 or np.isnan(pct_series.to_numpy()).all():
        mdd_pct = 0.0
    else:
        mdd_pct = float(np.nanmin(pct_series.to_numpy()))

    if dd_abs.isna().all():
        return 0.0, 0.0, None, None, pd.Timedelta(0)

    # Find drawdown start and end
    dd_end_idx = dd_abs.idxmax()
    dd_start_idx = equity.loc[:dd_end_idx].idxmax()

    dd_start = (
        timestamps.loc[dd_start_idx]
        if (timestamps is not None and dd_start_idx in timestamps.index)
        else dd_start_idx
    )
    dd_end = (
        timestamps.loc[dd_end_idx]
        if (timestamps is not None and dd_end_idx in timestamps.index)
        else dd_end_idx
    )

    dd_duration = pd.Timedelta(0)
    try:
        if isinstance(dd_start, pd.Timestamp) and isinstance(dd_end, pd.Timestamp):
            dd_duration = dd_end - dd_start
    except Exception:
        pass

    return mdd_abs, mdd_pct, dd_start, dd_end, dd_duration


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _as_float(x: Any) -> float:
    """Safely convert to float."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x: Any) -> int:
    """Safely convert to int."""
    try:
        return int(x)
    except Exception:
        return 0


def _timespan_days_from_index(idx: pd.DatetimeIndex) -> float:
    """Return total timespan in days from a datetime index."""
    if idx is None or len(idx) < 2:
        return 0.0
    delta = idx[-1] - idx[0]
    return float(delta.total_seconds()) / 86400.0


# ============================================================================
# ENHANCED POSITION TRACKING (LONG/SHORT SUPPORT)
# ============================================================================
def compute_open_positions(trades: pd.DataFrame) -> Dict[str, Tuple[int, float]]:
    """
    Reconstruct current open positions from trade history.
    Supports LONG and SHORT positions.
    
    Args:
        trades: DataFrame with columns: symbol, action, quantity, price, status, timestamp
    
    Returns:
        Dict mapping symbol to (signed_qty, avg_price)
        - signed_qty > 0 = LONG position
        - signed_qty < 0 = SHORT position
    """
    if trades is None or trades.empty:
        return {}

    df = trades.copy()

    # Ensure required columns exist
    for col in ("symbol", "action", "quantity", "price", "status", "timestamp"):
        if col not in df.columns:
            df[col] = np.nan

    # Filter to filled trades only
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "filled"]

    if df.empty:
        return {}

    df = df.sort_values("timestamp")

    # Track signed positions (positive = long, negative = short)
    position_map: Dict[str, int] = {}  # symbol -> signed_qty
    avg_map: Dict[str, float] = {}     # symbol -> avg_entry_price

    for _, row in df.iterrows():
        sym = str(row.get("symbol"))
        act = str(row.get("action", "")).upper()
        qty = _as_int(row.get("quantity"))
        px = _as_float(row.get("price"))

        if not np.isfinite(px) or qty <= 0 or not sym:
            continue

        current_pos = position_map.get(sym, 0)
        current_avg = avg_map.get(sym, 0.0)

        if act == "BUY":
            # Buying: increases position (or reduces short)
            new_pos = current_pos + qty
            
            if current_pos >= 0:
                # Already long or flat -> add to long
                if new_pos != 0:
                    avg_map[sym] = (current_avg * current_pos + px * qty) / new_pos
                else:
                    avg_map[sym] = 0.0
            else:
                # Currently short -> covering
                if new_pos >= 0:
                    # Fully covered or flipped to long
                    avg_map[sym] = px if new_pos > 0 else 0.0
                else:
                    # Still short, keep short avg
                    pass
                
            position_map[sym] = new_pos

        elif act == "SELL":
            # Selling: decreases position (or increases short)
            new_pos = current_pos - qty
            
            if current_pos <= 0:
                # Already short or flat -> add to short
                if new_pos != 0:
                    avg_map[sym] = (abs(current_avg * current_pos) + px * qty) / abs(new_pos)
                else:
                    avg_map[sym] = 0.0
            else:
                # Currently long -> reducing
                if new_pos <= 0:
                    # Fully closed or flipped to short
                    avg_map[sym] = px if new_pos < 0 else 0.0
                else:
                    # Still long, keep long avg
                    pass
                
            position_map[sym] = new_pos

    # Return only non-zero positions
    return {
        sym: (position_map[sym], avg_map[sym]) 
        for sym in position_map 
        if position_map[sym] != 0
    }


def compute_open_pl(
    trades: pd.DataFrame,
    price_lookup: Callable[[str], Optional[float]],
) -> tuple[float, list[dict]]:
    """
    Calculate open P/L from current positions and live prices.
    
    Args:
        trades: Trade history DataFrame
        price_lookup: Function that returns current price for a symbol
    
    Returns:
        Tuple of (total_open_pl, details_list)
    """
    open_pos = compute_open_positions(trades)
    total = 0.0
    rows = []

    for sym, (signed_qty, avg) in open_pos.items():
        last = price_lookup(sym) if callable(price_lookup) else None
        
        if last is None or not np.isfinite(last):
            mtm = 0.0
            last_out = None
        else:
            last_out = float(last)
            # P/L = (current_price - avg_price) * signed_qty
            # For LONG: positive qty, profit if price rises
            # For SHORT: negative qty, profit if price falls
            mtm = (last_out - float(avg)) * int(signed_qty)

        total += mtm

        rows.append({
            "symbol": sym,
            "qty": int(signed_qty),
            "direction": "LONG" if signed_qty > 0 else "SHORT",
            "avg": float(avg),
            "last": last_out,
            "open_pl": float(mtm),
        })

    return float(total), rows


# ============================================================================
# ENHANCED SHARPE/SORTINO WITH CONSERVATIVE ANNUALIZATION
# ============================================================================
def calculate_sharpe_sortino(
    equity_series: pd.Series,
    freq: str = "daily"
) -> tuple[float, float, str]:
    """
    Calculate Sharpe and Sortino ratios with automatic frequency detection.

    Conservative behavior (Option A):
      - If timespan < 2 days → no annualization (raw mean/std ratios)
      - For intraday annualization, cap factor at sqrt(252)
    
    Args:
        equity_series: Time-indexed equity curve
        freq: "daily" (default) or "intraday"
    
    Returns:
        (sharpe, sortino, actual_freq_used)
    """
    if equity_series is None or len(equity_series) < MIN_RETURNS_FOR_SHARPE:
        return 0.0, 0.0, "insufficient_data"

    equity_series = equity_series.dropna().sort_index()
    if equity_series.empty:
        return 0.0, 0.0, "insufficient_data"

    timespan_days = _timespan_days_from_index(equity_series.index)

    # Try daily first
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    
    if len(daily_returns) >= MIN_DAILY_RETURNS_FOR_SHARPE:
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        # Raw Sharpe if very short timespan
        if timespan_days < 2 or std_return <= 1e-6:
            sharpe = float(mean_return / std_return) if std_return > 1e-6 else 0.0
            # Sortino
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1:
                downside_std = downside_returns.std()
                if downside_std > 1e-6:
                    sortino = float(mean_return / downside_std)
                else:
                    sortino = 99.99 if mean_return > 0 else 0.0
            else:
                sortino = 99.99 if mean_return > 0 else 0.0
        else:
            # Annualized daily Sharpe
            sharpe = float(mean_return / std_return * np.sqrt(TRADING_DAYS_PER_YEAR))
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 1:
                downside_std = downside_returns.std()
                if downside_std > 1e-6:
                    sortino = float(
                        mean_return / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
                    )
                else:
                    sortino = 99.99 if mean_return > 0 else 0.0
            else:
                sortino = 99.99 if mean_return > 0 else 0.0

        return sharpe, sortino, "daily"

    # ---------------------------------------------------------------------
    # Fall back to intraday / per-trade returns
    # ---------------------------------------------------------------------
    trade_returns = equity_series.pct_change().dropna()
    if len(trade_returns) < MIN_RETURNS_FOR_SHARPE:
        return 0.0, 0.0, "insufficient_data"

    mean_return = trade_returns.mean()
    std_return = trade_returns.std()

    if std_return <= 1e-6:
        sharpe = 0.0
        downside_returns = trade_returns[trade_returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            sortino = float(mean_return / downside_std) if downside_std > 1e-6 else (
                99.99 if mean_return > 0 else 0.0
            )
        else:
            sortino = 99.99 if mean_return > 0 else 0.0
        return sharpe, sortino, "intraday"

    if timespan_days < 2:
        # No annualization for ultra-short periods
        sharpe = float(mean_return / std_return)
        downside_returns = trade_returns[trade_returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            if downside_std > 1e-6:
                sortino = float(mean_return / downside_std)
            else:
                sortino = 99.99 if mean_return > 0 else 0.0
        else:
            sortino = 99.99 if mean_return > 0 else 0.0
        return sharpe, sortino, "intraday"

    # Longer timespan: conservative intraday annualization
    time_span_minutes = (
        (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 60.0
    )
    trades_per_minute = len(trade_returns) / max(time_span_minutes, 1.0)
    trades_per_year = trades_per_minute * TRADING_MINUTES_PER_YEAR

    # Cap annualization factor at sqrt(252) for conservatism
    annualization_factor = float(
        np.sqrt(min(trades_per_year, TRADING_DAYS_PER_YEAR))
    )

    sharpe = float(mean_return / std_return * annualization_factor)

    downside_returns = trade_returns[trade_returns < 0]
    if len(downside_returns) > 1:
        downside_std = downside_returns.std()
        if downside_std > 1e-6:
            sortino = float(mean_return / downside_std * annualization_factor)
        else:
            sortino = 99.99 if mean_return > 0 else 0.0
    else:
        sortino = 99.99 if mean_return > 0 else 0.0

    return sharpe, sortino, "intraday"


# ============================================================================
# PROFESSIONAL PERFORMANCE METRICS
# ============================================================================
def calculate_metrics(
    df: pd.DataFrame,
    price_lookup: Optional[Callable[[str], Optional[float]]] = None,
    start_equity: float = 10000.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics.
    
    Metrics included:
    - Closed P/L, Win Rate, Profit Factor
    - Sharpe Ratio (daily or intraday, annualized conservatively)
    - Sortino Ratio (downside deviation only)
    - CAGR (Compound Annual Growth Rate, only if timespan >= 30 days)
    - Volatility (annualized, conservative)
    - Max Drawdown (absolute and percentage)
    - Trade statistics
    - Open P/L (if price_lookup provided)
    
    Args:
        df: Trade history DataFrame
        price_lookup: Optional function to get current prices for open positions
        start_equity: Starting account equity
    
    Returns:
        Dictionary of metrics
    """
    if df is None or df.empty:
        out = default_metrics()
        if price_lookup is not None:
            out["open_pl"] = 0.0
            out["open_pl_details"] = []
        out.update({
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe_daily": 0.0,
            "sortino": 0.0,
            "max_equity": float(start_equity),
            "min_equity": float(start_equity),
            "equity_end": float(start_equity),
            "metrics_warning": "",
        })
        return out

    df = df.copy()

    # Normalize columns
    if "pnl" not in df.columns:
        df["pnl"] = 0.0
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT

    df["pnl_clean"] = pd.to_numeric(df["pnl"], errors="coerce")

    # ========================================================================
    # IDENTIFY REALIZED TRADES (prefer filled trades with non-null PnL)
    # ========================================================================
    if {"action", "status"}.issubset(df.columns):
        s = df["status"].astype(str).str.lower()
        mask = s.eq("filled") & df["pnl_clean"].notna()
        realized_df = df[mask].copy()
        
        if realized_df.empty:
            # Fallback: all rows with non-null PnL
            realized_df = df[df["pnl_clean"].notna()].copy()
    else:
        realized_df = df[df["pnl_clean"].notna()].copy()

    realized = realized_df["pnl_clean"]

    # If no realized trades, return defaults
    if realized.empty:
        base = _calculate_base_metrics(df, realized, None, price_lookup, start_equity)
        base.update({
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe_daily": 0.0,
            "sortino": 0.0,
            "max_equity": float(start_equity),
            "min_equity": float(start_equity),
            "equity_end": float(start_equity),
            "metrics_warning": "",
        })
        return base

    # ========================================================================
    # BUILD EQUITY SERIES (fixed positional logic)
    # ========================================================================
    realized_ts = pd.to_datetime(realized_df["timestamp"], errors="coerce")

    equity_series = pd.Series(
        realized.cumsum().to_numpy() + float(start_equity),
        index=realized_ts,
        name="equity",
    )
    equity_series = equity_series.dropna().sort_index()

    if equity_series.empty:
        base = _calculate_base_metrics(df, realized, None, price_lookup, start_equity)
        base.update({
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe_daily": 0.0,
            "sortino": 0.0,
            "max_equity": float(start_equity),
            "min_equity": float(start_equity),
            "equity_end": float(start_equity),
            "metrics_warning": "",
        })
        return base

    timespan_days = _timespan_days_from_index(equity_series.index)

    # ========================================================================
    # ENHANCED SHARPE/SORTINO WITH AUTO FREQUENCY DETECTION
    # ========================================================================
    sharpe_daily, sortino, freq_used = calculate_sharpe_sortino(equity_series)
    
    # Generate warning message if using intraday or insufficient data
    if freq_used == "intraday":
        daily_count = len(equity_series.resample("1D").last().dropna())
        metrics_warning = (
            f"⚠️ Using intraday Sharpe/Sortino (only {daily_count} trading days). "
            f"Collect {MIN_DAILY_RETURNS_FOR_SHARPE}+ days for stable daily metrics."
        )
    elif freq_used == "insufficient_data":
        metrics_warning = "⚠️ Insufficient data for Sharpe/Sortino calculation."
    else:
        metrics_warning = ""

    # ========================================================================
    # DAILY RETURNS FOR VOLATILITY (CONSERVATIVE)
    # ========================================================================
    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    
    if len(daily_returns) > 1 and timespan_days >= 2:
        volatility = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        # Fall back to trade-level volatility
        trade_returns = equity_series.pct_change().dropna()
        if len(trade_returns) > 1:
            if timespan_days >= 2:
                time_span_minutes = (
                    (equity_series.index[-1] - equity_series.index[0]).total_seconds()
                    / 60.0
                )
                trades_per_minute = len(trade_returns) / max(time_span_minutes, 1.0)
                trades_per_year = trades_per_minute * TRADING_MINUTES_PER_YEAR
                annualization_factor = float(
                    np.sqrt(min(trades_per_year, TRADING_DAYS_PER_YEAR))
                )
                volatility = float(trade_returns.std() * annualization_factor)
            else:
                # No annualization for ultra-short samples
                volatility = float(trade_returns.std())
        else:
            volatility = 0.0

    # ========================================================================
    # CAGR (Compound Annual Growth Rate) – ONLY if timespan >= 30 days
    # ========================================================================
    if len(daily_equity) > 1 and timespan_days >= 30.0:
        start_val = float(daily_equity.iloc[0])
        end_val = float(daily_equity.iloc[-1])
        days = (daily_equity.index[-1] - daily_equity.index[0]).days
        years = max(days / 365.25, 1 / 365.25)  # At least 1 day
        
        if start_val > 0:
            cagr = float((end_val / start_val) ** (1.0 / years) - 1.0)
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    # ========================================================================
    # CALCULATE BASE METRICS
    # ========================================================================
    base = _calculate_base_metrics(df, realized, equity_series, price_lookup, start_equity)

    # ========================================================================
    # ADD ADVANCED METRICS
    # ========================================================================
    base.update({
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_daily": sharpe_daily,
        "sortino": sortino,
        "max_equity": float(equity_series.max()),
        "min_equity": float(equity_series.min()),
        "equity_end": float(equity_series.iloc[-1]),
        "metrics_warning": metrics_warning,
        "metrics_frequency": freq_used,  # "daily", "intraday", or "insufficient_data"
    })

    return base


# ============================================================================
# BASE METRICS CALCULATOR
# ============================================================================
def _calculate_base_metrics(
    df: pd.DataFrame,
    realized: pd.Series,
    equity_series: Optional[pd.Series],
    price_lookup: Optional[Callable[[str], Optional[float]]],
    start_equity: float,
) -> Dict[str, Any]:
    """
    Calculate base trading metrics (win rate, profit factor, drawdown, etc).
    
    Internal function called by calculate_metrics().
    """
    realized = realized.dropna()
    wins = realized[realized > 0]
    losses = realized[realized < 0]
    realized_nz = realized[realized != 0]
    trade_count = int(len(realized_nz))

    # ========================================================================
    # MAX DRAWDOWN (on equity if available)
    # ========================================================================
    if equity_series is not None and not equity_series.empty:
        pnl_for_mdd = equity_series.diff().fillna(0.0)
        ts_for_mdd = equity_series.index.to_series()
        start_eq_for_mdd = float(equity_series.iloc[0])
        mdd_abs, mdd_pct, dd_start, dd_end, dd_dur = max_drawdown_stats(
            pnl_for_mdd,
            timestamps=ts_for_mdd,
            start_equity=start_eq_for_mdd,
        )
    else:
        timestamps = df["timestamp"] if "timestamp" in df.columns else None
        mdd_abs, mdd_pct, dd_start, dd_end, dd_dur = max_drawdown_stats(
            realized,
            timestamps=timestamps,
            start_equity=0.0,
        )

    # ========================================================================
    # WIN/LOSS STATISTICS
    # ========================================================================
    win_count = int(len(wins))
    loss_count = int(len(losses))
    avg_win = float(wins.mean()) if win_count else 0.0
    avg_loss = float(losses.mean()) if loss_count else 0.0

    # ========================================================================
    # AVERAGE TRADE DURATION
    # ========================================================================
    avg_duration = 0.0
    if "duration" in df.columns:
        dur = pd.to_numeric(df["duration"], errors="coerce").dropna()
        avg_duration = float(dur.mean()) if not dur.empty else 0.0

    # ========================================================================
    # PROFIT FACTOR (robust)
    # ========================================================================
    total_win = float(wins.sum())
    total_loss = float(abs(losses.sum()))
    
    if total_loss > 0.0:
        profit_factor = total_win / total_loss
    else:
        profit_factor = float("inf") if total_win > 0.0 else 0.0
    
    # Limit to reasonable display range
    if np.isinf(profit_factor):
        profit_factor = 999.99

    # ========================================================================
    # ASSEMBLE RESULTS
    # ========================================================================
    result = {
        "wins": win_count,
        "losses": loss_count,
        "win_rate": float((win_count / max(1, trade_count)) * 100.0),
        "closed_pl": float(realized.sum()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
        "max_drawdown": float(mdd_abs),
        "max_drawdown_pct": float(mdd_pct * 100.0),
        "max_dd_start": dd_start if dd_start is not None else "",
        "max_dd_end": dd_end if dd_end is not None else "",
        "max_dd_duration": float(dd_dur.total_seconds()),
        "avg_trade_duration": avg_duration,
        "total_pnl": float(realized.sum()),
        "number_of_trades": int(trade_count),
    }

    # ========================================================================
    # OPEN P/L (if price lookup provided)
    # ========================================================================
    if price_lookup is not None:
        open_pl_total, open_rows = compute_open_pl(df, price_lookup)
        result["open_pl"] = float(open_pl_total)
        result["open_pl_details"] = open_rows
    else:
        result["open_pl"] = 0.0
        result["open_pl_details"] = []

    return result


# ============================================================================
# DEFAULT METRICS (for empty data)
# ============================================================================
def default_metrics(n: int = 0) -> Dict[str, Any]:
    """Return default metrics structure with zero values."""
    return {
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "closed_pl": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "max_dd_start": "",
        "max_dd_end": "",
        "max_dd_duration": 0.0,
        "avg_trade_duration": 0.0,
        "total_pnl": 0.0,
        "number_of_trades": int(n),
        "open_pl": 0.0,
        "open_pl_details": [],
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe_daily": 0.0,
        "sortino": 0.0,
        "max_equity": 0.0,
        "min_equity": 0.0,
        "equity_end": 0.0,
        "metrics_warning": "",
        "metrics_frequency": "",
    }
