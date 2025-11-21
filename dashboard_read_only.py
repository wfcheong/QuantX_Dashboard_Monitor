# dashboard_read_only.py
"""
Read-only Streamlit dashboard for monitoring trading strategies.
Displays trade logs and performance metrics without execution control.

Features:
- Auto-discovery of strategy logs
- Real-time performance metrics (Sharpe, Sortino, CAGR, etc.)
- Live prices from IBKR for Open P/L calculation
- Interactive charts and visualizations
- Date range filtering
- Export capabilities
"""

import os
import sys
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache
import time as time_module
import threading
import asyncio

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

try:
    import altair as alt
    _ALT_AVAILABLE = True
except Exception:
    _ALT_AVAILABLE = False

try:
    from ib_insync import IB, Stock, util
    _IB_AVAILABLE = True
except Exception:
    _IB_AVAILABLE = False

# import inspect
# import importlib
# import utils.performance_metrics
# importlib.reload(utils.performance_metrics)

from dotenv import load_dotenv
# from utils.performance_metrics import calculate_metrics
from utils.client_id_manager import get_or_allocate_client_id

# Use:
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.performance_metrics import calculate_metrics, calculate_sharpe_sortino

# Verify it loaded
print(f"✅ Loaded calculate_sharpe_sortino: {callable(calculate_sharpe_sortino)}")

# Check if enhanced version is loaded
# source = inspect.getsource(calculate_metrics)
# has_enhanced = "calculate_sharpe_sortino" in source

st.sidebar.write("---")
# st.sidebar.subheader("🔬 Debug Info")
# st.sidebar.caption(f"Enhanced metrics: {'✅ YES' if has_enhanced else '❌ NO'}")
# st.sidebar.caption(f"Module location: {inspect.getfile(calculate_metrics)}")

# if not has_enhanced:
#     st.sidebar.error("⚠️ Old metrics file detected! Replace with enhanced version.")
# st.sidebar.write("---")

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()

START_EQUITY = float(os.getenv("START_EQUITY", "10000"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "-1000"))
MAX_TOTAL_DRAWDOWN = float(os.getenv("MAX_TOTAL_DRAWDOWN", "-5000"))
APP_ENV = os.getenv("APP_ENV", "local").upper()
DASHBOARD_TIMEZONE = os.getenv("DASHBOARD_TIMEZONE", "US/Eastern")
LOG_ROOT = os.getenv("LOG_ROOT", "strategies_runner/logs")

# IBKR Connection settings
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
ENABLE_LIVE_PRICES = os.getenv("ENABLE_LIVE_PRICES", "true").lower() == "true"

st.set_page_config(
    page_title="Quant X Trading Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_PATH = os.path.join("assets", "logo.png")
US_ET = ZoneInfo("America/New_York")

EXPECTED_COLUMNS = [
    "timestamp", "symbol", "action", "price", "quantity",
    "pnl", "duration", "position", "status", "ib_order_id"
]


# ============================================================================
# EVENT LOOP SETUP FOR IB_INSYNC
# ============================================================================
def _ensure_event_loop():
    """Ensure asyncio event loop exists for ib_insync."""
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


_ensure_event_loop()


# ============================================================================
# IBKR PRICE LOOKUP (Thread-safe, Cached)
# ============================================================================
class IBKRPriceFetcher:
    """
    Thread-safe IBKR price fetcher with caching and connection management.
    
    Features:
    - Dedicated client ID for dashboard
    - Connection pooling/reuse
    - 60-second price caching
    - Automatic reconnection on failures
    - Thread-safe operations
    """
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ib: Optional[IB] = None
        self.client_id: Optional[int] = None
        self.lock = threading.Lock()
        self.price_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self.cache_ttl = 60  # seconds
        self.connection_failures = 0
        self.max_failures = 3
        
        # Suppress ib_insync logging
        util.logToConsole = False
    
    def _connect(self) -> bool:
        """Establish connection to IBKR."""
        try:
            if self.ib is None:
                self.ib = IB()
            
            if not self.ib.isConnected():
                # Get dedicated client ID for dashboard
                if self.client_id is None:
                    self.client_id = get_or_allocate_client_id(
                        name="dashboard_price_feed",
                        role="dashboard"
                    )
                
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.connection_failures = 0
                return True
            
            return True
            
        except Exception as e:
            self.connection_failures += 1
            if self.connection_failures <= self.max_failures:
                st.warning(f"⚠️ IBKR connection failed (attempt {self.connection_failures}/{self.max_failures}): {e}")
            return False
    
    def _disconnect(self):
        """Safely disconnect from IBKR."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
        except Exception:
            pass
    
    def _get_price_from_cache(self, symbol: str) -> Optional[float]:
        """Get price from cache if fresh."""
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            age = time_module.time() - timestamp
            if age < self.cache_ttl:
                return price
        return None
    
    def _fetch_live_price(self, symbol: str) -> Optional[float]:
        """Fetch live price from IBKR (blocking)."""
        try:
            # Create contract
            contract = Stock(symbol, "SMART", "USD")
            
            # Request market data
            self.ib.reqMktData(contract, "", False, False)
            
            # Wait for tick data (with timeout)
            for _ in range(20):  # 2 second timeout (20 * 0.1s)
                self.ib.sleep(0.1)
                ticker = self.ib.ticker(contract)
                
                # Try multiple price sources in priority order
                if ticker.last and np.isfinite(ticker.last):
                    price = float(ticker.last)
                    break
                elif ticker.close and np.isfinite(ticker.close):
                    price = float(ticker.close)
                    break
                elif ticker.bid and ticker.ask:
                    if np.isfinite(ticker.bid) and np.isfinite(ticker.ask):
                        price = float((ticker.bid + ticker.ask) / 2)
                        break
            else:
                # Timeout - no valid price received
                return None
            
            # Cancel market data to avoid subscription fees
            self.ib.cancelMktData(contract)
            
            # Cache the price
            self.price_cache[symbol] = (price, time_module.time())
            
            return price
            
        except Exception as e:
            # Silent fail - will use fallback
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for symbol (thread-safe).
        
        Returns cached price if fresh, otherwise fetches from IBKR.
        """
        with self.lock:
            # Check cache first
            cached = self._get_price_from_cache(symbol)
            if cached is not None:
                return cached
            
            # Connect if needed
            if not self._connect():
                return None
            
            # Fetch live price
            return self._fetch_live_price(symbol)
    
    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch prices for multiple symbols efficiently.
        
        Returns dict of symbol -> price.
        """
        prices = {}
        
        with self.lock:
            # Connect once for batch
            if not self._connect():
                return prices
            
            for symbol in symbols:
                # Check cache
                cached = self._get_price_from_cache(symbol)
                if cached is not None:
                    prices[symbol] = cached
                    continue
                
                # Fetch live
                price = self._fetch_live_price(symbol)
                if price is not None:
                    prices[symbol] = price
        
        return prices
    
    def clear_cache(self):
        """Clear price cache (e.g., on manual refresh)."""
        with self.lock:
            self.price_cache.clear()
    
    def close(self):
        """Clean shutdown."""
        with self.lock:
            self._disconnect()


# ============================================================================
# GLOBAL PRICE FETCHER INSTANCE
# ============================================================================
@st.cache_resource
def get_price_fetcher() -> Optional[IBKRPriceFetcher]:
    """Get singleton IBKR price fetcher."""
    if not _IB_AVAILABLE:
        return None
    
    try:
        fetcher = IBKRPriceFetcher(IB_HOST, IB_PORT)
        return fetcher
    except Exception as e:
        st.error(f"❌ Failed to initialize IBKR price fetcher: {e}")
        return None


# ============================================================================
# PRICE LOOKUP FUNCTIONS
# ============================================================================
def create_ibkr_price_lookup(fetcher: IBKRPriceFetcher) -> Callable[[str], Optional[float]]:
    """Create price lookup function using IBKR."""
    def lookup(symbol: str) -> Optional[float]:
        return fetcher.get_price(symbol)
    return lookup


def create_last_trade_price_lookup(df: pd.DataFrame) -> Callable[[str], Optional[float]]:
    """
    Fallback price lookup using last trade price from log.
    Used when live prices are unavailable.
    """
    def lookup(symbol: str) -> Optional[float]:
        if df.empty or "symbol" not in df.columns:
            return None
        
        symbol_trades = df[df["symbol"] == symbol]
        if symbol_trades.empty:
            return None
        
        # Get last filled trade
        filled = symbol_trades[
            symbol_trades["status"].astype(str).str.lower() == "filled"
        ]
        
        if filled.empty:
            return None
        
        last_row = filled.sort_values("timestamp", ascending=False).iloc[0]
        price = last_row.get("price")
        
        if pd.notna(price):
            return float(price)
        
        return None
    
    return lookup


# ============================================================================
# STRATEGY LOG DISCOVERY
# ============================================================================
def discover_strategy_logs(log_root: str) -> Dict[str, str]:
    """
    Auto-discover strategy logs.
    
    Expected structure:
        LOG_ROOT/
            StrategyName/
                trade_log.csv
    
    Returns:
        Dict mapping strategy_name -> trade_log_path
    """
    strategies: Dict[str, str] = {}

    if not os.path.isdir(log_root):
        return strategies

    for name in sorted(os.listdir(log_root)):
        strategy_dir = os.path.join(log_root, name)
        if not os.path.isdir(strategy_dir):
            continue

        trade_log_path = os.path.join(strategy_dir, "trade_log.csv")
        if os.path.exists(trade_log_path):
            strategies[name] = trade_log_path

    return strategies


# ============================================================================
# TRADE LOG LOADER
# ============================================================================
def safe_numeric_convert(series: pd.Series, col_name: str) -> pd.Series:
    """Convert column to numeric with error reporting."""
    cleaned = series.astype(str).str.replace("[^0-9eE+\\-.]", "", regex=True)
    result = pd.to_numeric(cleaned, errors="coerce")
    
    nan_count = result.isna().sum()
    if nan_count > 0 and nan_count < len(result):
        st.warning(f"⚠️ {col_name}: {nan_count}/{len(result)} values could not be parsed")
    
    return result


def load_trade_log(path: str) -> pd.DataFrame:
    """
    Load and validate trade log CSV.
    
    Returns:
        Clean DataFrame with standardized columns and types
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)
    except Exception as e:
        st.error(f"❌ Failed to load: {path}")
        st.caption(f"Error: {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = df.copy()

    # Check for required columns
    required_cols = ["timestamp", "symbol", "action", "price", "quantity", "pnl", "status"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"❌ Trade log missing required columns: {missing}")
        st.caption(f"File: {path}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Ensure all expected columns exist
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    # Parse timestamp (keep timezone-aware for consistency)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    
    # Remove rows with invalid timestamps
    invalid_ts = df["timestamp"].isna().sum()
    if invalid_ts > 0:
        st.warning(f"⚠️ Removed {invalid_ts} rows with invalid timestamps")
        df = df.dropna(subset=["timestamp"])

    # Safe numeric conversions
    df["price"] = safe_numeric_convert(df["price"], "price")
    df["pnl"] = safe_numeric_convert(df["pnl"], "pnl")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _local_now() -> datetime:
    """Get current local time."""
    return datetime.now().astimezone()


def _format_local_tz(dt: datetime) -> str:
    """Format timezone offset as UTC±HH:MM."""
    offset = dt.utcoffset() or timedelta(0)
    sign = "+" if offset >= timedelta(0) else "-"
    minutes = int(abs(offset.total_seconds()) // 60)
    hh, mm = divmod(minutes, 60)
    return f"UTC{sign}{hh:02d}:{mm:02d}"


def _format_td(delta: timedelta) -> str:
    """Format timedelta as 'Xh Ym'."""
    secs = int(max(0, delta.total_seconds()))
    h, r = divmod(secs, 3600)
    m, _ = divmod(r, 60)
    return f"{h}h {m}m"


def _us_market_status() -> str:
    """Return US market status string."""
    now_et = _local_now().astimezone(US_ET)
    weekday = now_et.weekday()
    t = now_et.time()
    
    open_t = dtime(9, 30)
    close_t = dtime(16, 0)
    
    # Market open
    if 0 <= weekday <= 4 and open_t <= t <= close_t:
        closes_at = datetime.combine(now_et.date(), close_t, tzinfo=US_ET)
        left = closes_at - now_et
        return f"🟢 US Market: OPEN (closes in {_format_td(left)})"
    
    # Before open today
    if 0 <= weekday <= 4 and t < open_t:
        opens_at = datetime.combine(now_et.date(), open_t, tzinfo=US_ET)
        until = opens_at - now_et
        return f"🔴 US Market: CLOSED (opens in {_format_td(until)})"
    
    # After close or weekend
    days = 1
    next_date = (now_et + timedelta(days=days)).date()
    while next_date.weekday() > 4:
        days += 1
        next_date = (now_et + timedelta(days=days)).date()
    
    next_open = datetime.combine(next_date, open_t, tzinfo=US_ET)
    until = next_open - now_et
    return f"🔴 US Market: CLOSED (opens in {_format_td(until)})"


def try_load_logo(path: str, width: int = 100):
    """Load and display logo if available."""
    if not _PIL_AVAILABLE:
        return
    
    if os.path.exists(path):
        try:
            st.image(Image.open(path), width=width)
        except Exception:
            pass


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Settings")
st.sidebar.caption(f"**Environment:** `{APP_ENV}`")
st.sidebar.caption(f"**Log Root:** `{LOG_ROOT}`")

st.sidebar.subheader("⏰ Market Clock")
st.sidebar.caption(_us_market_status())

now = _local_now()
st.sidebar.caption(
    f"**Local Time:** {now:%Y-%m-%d %H:%M:%S}  \n"
    f"**Timezone:** {_format_local_tz(now)}"
)
st.sidebar.caption("*Trade logs are in UTC*")

# IBKR Connection Status
st.sidebar.subheader("📡 IBKR Connection")
if _IB_AVAILABLE:
    st.sidebar.caption(f"**Host:** {IB_HOST}:{IB_PORT}")
    
    price_fetcher = get_price_fetcher()
    
    if price_fetcher:
        # Test connection
        try:
            with price_fetcher.lock:
                is_connected = price_fetcher._connect()
            
            if is_connected:
                st.sidebar.success("✅ Connected to IBKR")
                st.sidebar.caption(f"Client ID: {price_fetcher.client_id}")
            else:
                st.sidebar.error("❌ Not connected")
                ENABLE_LIVE_PRICES = False
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {e}")
            ENABLE_LIVE_PRICES = False
    else:
        st.sidebar.error("❌ Price fetcher unavailable")
        ENABLE_LIVE_PRICES = False
else:
    st.sidebar.warning("⚠️ ib_insync not available")
    ENABLE_LIVE_PRICES = False
    price_fetcher = None

# Live prices toggle
st.sidebar.subheader("💹 Live Prices")
if _IB_AVAILABLE and price_fetcher:
    ENABLE_LIVE_PRICES = st.sidebar.checkbox(
        "Enable live price lookup",
        value=ENABLE_LIVE_PRICES,
        help="Fetch real-time prices from IBKR for Open P/L calculation"
    )
    
    if ENABLE_LIVE_PRICES:
        st.sidebar.info("📊 Using IBKR live market data")
        
        # Cache clear button
        if st.sidebar.button("🔄 Clear Price Cache", help="Force refresh all prices"):
            if price_fetcher:
                price_fetcher.clear_cache()
                st.sidebar.success("Cache cleared!")
    else:
        st.sidebar.info("📝 Using last trade prices from logs")
else:
    st.sidebar.warning("Live prices disabled")

# Refresh controls
st.sidebar.subheader("🔄 Refresh")
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    if price_fetcher:
        price_fetcher.clear_cache()
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time_module.sleep(30)
    st.rerun()

# Date filter
st.sidebar.subheader("📅 Date Filter")
use_date_filter = st.sidebar.checkbox("Enable date filter", value=False)

if use_date_filter:
    date_from = st.sidebar.date_input(
        "From",
        value=datetime.now() - timedelta(days=30)
    )
    date_to = st.sidebar.date_input(
        "To",
        value=datetime.now()
    )
else:
    date_from = None
    date_to = None


# ============================================================================
# DISCOVER STRATEGIES
# ============================================================================
strategies = discover_strategy_logs(LOG_ROOT)

if not strategies:
    st.error("❌ No strategy logs found")
    st.info(
        f"📁 Expected structure:\n\n"
        f"```\n"
        f"{LOG_ROOT}/\n"
        f"  StrategyName/\n"
        f"    trade_log.csv\n"
        f"```"
    )
    st.stop()

st.sidebar.success(f"✅ Found {len(strategies)} strategies")


# ============================================================================
# EXTRACT UNIQUE SYMBOLS FROM ALL STRATEGIES
# ============================================================================
def extract_all_symbols(strategy_frames: Dict[str, pd.DataFrame]) -> List[str]:
    """Extract unique symbols from all strategy dataframes."""
    symbols = set()
    for df in strategy_frames.values():
        if not df.empty and "symbol" in df.columns:
            symbols.update(df["symbol"].dropna().unique().tolist())
    return sorted(list(symbols))


# ============================================================================
# LOAD TRADE LOGS & CALCULATE METRICS
# ============================================================================
strategy_frames: Dict[str, pd.DataFrame] = {}
metrics_by_strategy: Dict[str, Dict[str, Any]] = {}
summary_rows = []

total_closed = 0.0
total_trades = 0
today_utc = pd.Timestamp.utcnow().normalize()
todays_pnl = 0.0

# Progress bar for loading
progress_bar = st.progress(0)
status_text = st.empty()

# First pass: Load all trade logs
for idx, (name, path) in enumerate(strategies.items()):
    status_text.text(f"Loading {name}...")
    progress_bar.progress((idx + 1) / len(strategies))
    
    # Load trade log
    df = load_trade_log(path)
    
    # Apply date filter if enabled
    if use_date_filter and not df.empty and date_from and date_to:
        df = df[
            (df["timestamp"].dt.date >= date_from) &
            (df["timestamp"].dt.date <= date_to)
        ]
    
    strategy_frames[name] = df

# Extract all unique symbols for batch price fetching
all_symbols = extract_all_symbols(strategy_frames)

# Batch fetch live prices if enabled
if ENABLE_LIVE_PRICES and price_fetcher and all_symbols:
    status_text.text(f"Fetching live prices for {len(all_symbols)} symbols...")
    live_prices = price_fetcher.get_prices_batch(all_symbols)
    st.sidebar.caption(f"💹 Loaded {len(live_prices)}/{len(all_symbols)} live prices")
else:
    live_prices = {}

# Second pass: Calculate metrics with prices
for idx, (name, df) in enumerate(strategy_frames.items()):
    status_text.text(f"Calculating metrics for {name}...")
    progress_bar.progress((idx + 1) / len(strategies))
    
    # Calculate metrics with appropriate price lookup
    if not df.empty:
        try:
            # Create price lookup function
            if ENABLE_LIVE_PRICES and price_fetcher and live_prices:
                # Use pre-fetched live prices
                def create_cached_lookup(prices_dict):
                    def lookup(symbol: str) -> Optional[float]:
                        return prices_dict.get(symbol)
                    return lookup
                
                price_lookup = create_cached_lookup(live_prices)
            else:
                # Fallback to last trade price
                price_lookup = create_last_trade_price_lookup(df)
            
            m = calculate_metrics(
                df,
                price_lookup=price_lookup,
                start_equity=START_EQUITY
            )
        except Exception as e:
            st.error(f"❌ Metrics calculation failed for {name}")
            st.exception(e)
            m = {}
    else:
        m = {}

    metrics_by_strategy[name] = m

    # Aggregate portfolio metrics
    closed = float(m.get("closed_pl", 0.0))
    total_closed += closed
    total_trades += int(m.get("number_of_trades", 0))

    # Today's P/L
    if not df.empty and "timestamp" in df.columns:
        todays_df = df[df["timestamp"] >= today_utc]
        todays_pnl += pd.to_numeric(
            todays_df.get("pnl", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0).sum()

    # Get client ID (informational)
    try:
        client_id = get_or_allocate_client_id(name=name, role="strategy")
    except Exception:
        client_id = "N/A"

    # Build summary row
    summary_rows.append({
        "Strategy": name,
        "Client ID": client_id,
        "Closed P/L": round(closed, 2),
        "Open P/L": round(float(m.get("open_pl", 0.0)), 2),
        "Total P/L": round(closed + float(m.get("open_pl", 0.0)), 2),
        "Win Rate (%)": round(m.get("win_rate", 0.0), 2),
        "Wins": int(m.get("wins", 0)),
        "Losses": int(m.get("losses", 0)),
        "Profit Factor": round(m.get("profit_factor", 0.0), 2),
        "Sharpe (daily)": round(m.get("sharpe_daily", 0.0), 2),
        "Sortino": round(m.get("sortino", 0.0), 2),
        "CAGR (%)": round(m.get("cagr", 0.0) * 100.0, 2),
        "Volatility (%)": round(m.get("volatility", 0.0) * 100.0, 2),
        "Max DD": round(m.get("max_drawdown", 0.0), 2),
        "Max DD (%)": round(m.get("max_drawdown_pct", 0.0), 2),
        "Trades": int(m.get("number_of_trades", 0)),
    })

# Clear progress indicators
progress_bar.empty()
status_text.empty()


# ============================================================================
# HEADER
# ============================================================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    try_load_logo(LOGO_PATH, width=100)
with col_title:
    st.title("📊 Quant X Trading Monitor")
    st.caption("Read-only dashboard - Live monitoring with IBKR market data")


# ============================================================================
# PORTFOLIO OVERVIEW
# ============================================================================
st.subheader("💼 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Closed P&L",
    f"${total_closed:,.2f}",
    help="Total realized profit/loss from closed trades"
)

col2.metric(
    "Today's P&L",
    f"${todays_pnl:,.2f}",
    delta=f"{todays_pnl:,.2f}",
    help="Profit/loss for today (UTC)"
)

col3.metric(
    "Total Trades",
    f"{total_trades:,}",
    help="Total number of completed trades"
)

col4.metric(
    "Active Strategies",
    f"{len(strategies)}",
    help="Number of strategies with logs"
)

# Risk alerts
if todays_pnl <= MAX_DAILY_LOSS:
    st.error(
        f"🚨 **DAILY LOSS LIMIT EXCEEDED**  \n"
        f"Today's P/L: ${todays_pnl:.2f} ≤ Limit: ${MAX_DAILY_LOSS:.2f}"
    )

if total_closed <= MAX_TOTAL_DRAWDOWN:
    st.error(
        f"🚨 **TOTAL DRAWDOWN LIMIT EXCEEDED**  \n"
        f"Total P/L: ${total_closed:.2f} ≤ Limit: ${MAX_TOTAL_DRAWDOWN:.2f}"
    )


# ============================================================================
# PERFORMANCE SUMMARY TABLE
# ============================================================================
st.subheader("📊 Performance Summary")

if not summary_rows:
    st.info("No trade data available")
    st.stop()

summary_df = pd.DataFrame(summary_rows)

# Filter and sort controls
with st.expander("🔍 Filter & Sort Options", expanded=False):
    colf1, colf2 = st.columns(2)

    strategies_available = summary_df["Strategy"].unique().tolist()
    chosen = colf1.multiselect(
        "Select Strategies",
        options=strategies_available,
        default=strategies_available,
    )

    metric_options = [
        c for c in summary_df.columns 
        if c not in ("Strategy", "Client ID")
    ]
    default_sort_idx = (
        metric_options.index("Total P/L") 
        if "Total P/L" in metric_options 
        else 0
    )

    sort_by = colf2.selectbox(
        "Sort by Metric",
        options=metric_options,
        index=default_sort_idx,
    )
    ascending = colf2.checkbox("Sort Ascending", value=False)

# Apply filters
filtered_df = summary_df[summary_df["Strategy"].isin(chosen)].sort_values(
    by=sort_by,
    ascending=ascending,
)

# Display table
st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Closed P/L": st.column_config.NumberColumn(format="$%.2f"),
        "Open P/L": st.column_config.NumberColumn(format="$%.2f"),
        "Total P/L": st.column_config.NumberColumn(format="$%.2f"),
        "Win Rate (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
        "Sharpe (daily)": st.column_config.NumberColumn(format="%.2f"),
        "Sortino": st.column_config.NumberColumn(format="%.2f"),
        "CAGR (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Max DD": st.column_config.NumberColumn(format="$%.2f"),
        "Max DD (%)": st.column_config.NumberColumn(format="%.2f%%"),
    }
)

# Download button
st.download_button(
    "⬇️ Download Summary CSV",
    filtered_df.to_csv(index=False).encode("utf-8"),
    f"strategy_summary_{datetime.now():%Y%m%d_%H%M%S}.csv",
    "text/csv",
    use_container_width=False
)


# ============================================================================
# DETAILED STRATEGY VIEWS
# ============================================================================
st.subheader("📂 Detailed Strategy Logs")

for name, df in strategy_frames.items():
    with st.expander(f"📈 {name}", expanded=False):
        
        m = metrics_by_strategy.get(name, {}) or {}
        
        # Metrics row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        k1.metric(
            "Closed P/L",
            f"${float(m.get('closed_pl', 0.0)):,.2f}",
            help="Realized profit/loss from closed trades"
        )
        
        k2.metric(
            "Open P/L",
            f"${float(m.get('open_pl', 0.0)):,.2f}",
            help="Unrealized P/L from open positions (mark-to-market via IBKR)" if ENABLE_LIVE_PRICES else "Unrealized P/L (using last trade prices)"
        )
        
        k3.metric(
            "Win Rate",
            f"{float(m.get('win_rate', 0.0)):.1f}%",
            help="Percentage of winning trades"
        )
        
        k4.metric(
            "Profit Factor",
            f"{float(m.get('profit_factor', 0.0)):,.2f}",
            help="Gross profit / Gross loss (>1.0 is profitable)"
        )
        
        k5.metric(
            "Sharpe (daily)",
            f"{float(m.get('sharpe_daily', 0.0)):,.2f}",
            help="Risk-adjusted return (annualized, >1.0 good, >2.0 excellent)"
        )
        
        k6.metric(
            "Sortino",
            f"{float(m.get('sortino', 0.0)):,.2f}",
            help="Like Sharpe but only penalizes downside volatility (>1.5 good)"
        )
        
        # Additional metrics row
        k7, k8, k9, k10 = st.columns(4)
        
        k7.metric(
            "CAGR",
            f"{float(m.get('cagr', 0.0)) * 100.0:.2f}%",
            help="Compound Annual Growth Rate"
        )
        
        k8.metric(
            "Volatility",
            f"{float(m.get('volatility', 0.0)) * 100.0:.2f}%",
            help="Annualized volatility (standard deviation of returns)"
        )
        
        k9.metric(
            "Max Drawdown",
            f"${float(m.get('max_drawdown', 0.0)):,.2f}",
            help="Largest peak-to-trough decline"
        )
        
        k10.metric(
            "Trades",
            f"{int(m.get('number_of_trades', 0)):,}",
            help="Total number of completed trades"
        )
        
        st.write("---")

        # Display metrics warnings
        warning = m.get("metrics_warning", "")
        freq = m.get("metrics_frequency", "")

        if warning:
            st.warning(warning)
        elif freq == "intraday":
            st.info(
                "ℹ️ **Using Intraday Metrics**: Sharpe/Sortino calculated from trade-level returns. "
                "Collect 10+ trading days for more accurate daily metrics."
            )
        elif freq == "daily":
            st.success("✅ Using daily return metrics (recommended)")

        # Show data summary
        if not df.empty and "timestamp" in df.columns:
            trading_days = len(df["timestamp"].dt.date.unique())
            total_trades = len(df)
            
            st.caption(
                f"📊 Data: {total_trades:,} trades across {trading_days} trading days | "
                f"Avg {total_trades/max(trading_days, 1):.1f} trades/day"
            )

        st.write("---")
        
        # Trade log table
        st.write("### 📋 Trade Log")
        
        if df.empty:
            st.info("ℹ️ No trades recorded yet")
            continue
        
        # Show recent trades (last 100)
        display_limit = 100
        display_df = df.tail(display_limit).sort_values("timestamp", ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp (UTC)",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "price": st.column_config.NumberColumn(format="$%.2f"),
                "pnl": st.column_config.NumberColumn(format="$%.2f"),
            }
        )
        
        if len(df) > display_limit:
            st.caption(f"📊 Showing last {display_limit} of {len(df):,} total trades")
        
        # Download button
        st.download_button(
            f"⬇️ Download {name} Trades",
            df.to_csv(index=False).encode("utf-8"),
            f"{name}_trades_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "text/csv",
            key=f"dl_{name}"
        )
        
        # Charts
        st.write("### 📈 Performance Charts")
        
        if "pnl" in df.columns and not df.empty:
            chart_df = df.copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], utc=True)
            
            # Equity curve
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Equity Curve**")
                eq = chart_df[["timestamp", "pnl"]].copy()
                eq["cumulative_pnl"] = eq["pnl"].cumsum()
                st.line_chart(
                    eq.set_index("timestamp")["cumulative_pnl"],
                    use_container_width=True
                )
            
            with col_chart2:
                st.write("**Drawdown**")
                dd = eq.copy()
                dd["drawdown"] = dd["cumulative_pnl"].cummax() - dd["cumulative_pnl"]
                st.area_chart(
                    dd.set_index("timestamp")["drawdown"],
                    use_container_width=True,
                    color="#FF4B4B"
                )
        
        # Open positions detail
        if m.get("open_pl_details") and len(m["open_pl_details"]) > 0:
            st.write("### 💼 Open Positions")
            
            if ENABLE_LIVE_PRICES:
                st.caption("📊 Showing mark-to-market P/L using IBKR live prices")
            else:
                st.caption("📝 Showing P/L using last trade prices from log")
            
            open_pos_df = pd.DataFrame(m["open_pl_details"])
            st.dataframe(
                open_pos_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "qty": st.column_config.NumberColumn("Quantity"),
                    "direction": st.column_config.TextColumn("Direction"),
                    "avg": st.column_config.NumberColumn("Avg Entry", format="$%.2f"),
                    "last": st.column_config.NumberColumn("Last Price", format="$%.2f"),
                    "open_pl": st.column_config.NumberColumn("Open P/L", format="$%.2f"),
                }
            )
        
        # Advanced metrics (collapsible)
        with st.expander("🔬 Advanced Metrics (JSON)", expanded=False):
            st.json(m)


# ============================================================================
# FOOTER
# ============================================================================
st.write("---")

footer_cols = st.columns([2, 1])
with footer_cols[0]:
    st.caption(
        f"📊 Dashboard v2.0 | Environment: {APP_ENV} | "
        f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )
with footer_cols[1]:
    if ENABLE_LIVE_PRICES and price_fetcher:
        st.caption("💹 Live IBKR prices enabled")
    else:
        st.caption("📝 Using log prices")


# ============================================================================
# CLEANUP ON EXIT
# ============================================================================
def cleanup():
    """Clean shutdown of IBKR connection."""
    if price_fetcher:
        try:
            price_fetcher.close()
        except Exception:
            pass

import atexit

atexit.register(cleanup)
