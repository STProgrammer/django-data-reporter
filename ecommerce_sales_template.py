# --- cell ---
# ===================== E-commerce sales — Robust Cleaner (Audited) =====================
# Input : ecommerce_raw.csv  (arbitrary schema; typical cols may include order_date, order_id, unit_price, qty, revenue, product, category, country, customer_id, etc.)
# Output: ecommerce_clean.csv
#
# Goals:
#  - Robust datetime parsing (epoch + best-format + fallback) with audit
#  - Safe numeric coercion (currency/thousands handled) without nuking categories/IDs
#  - Light imputations for numeric columns (median), duplicates dropped
#
# Requirements: Python 3.9+, pandas>=2.2, numpy, matplotlib (plots elsewhere)



import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
import re

plt.rcParams['figure.figsize'] = (8,4)

RAW_PATH = "ecommerce_raw.csv"      # replace with client file
CLEAN_PATH = "ecommerce_clean.csv"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def save_chart(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
# --- cell ---
# ---------- Helpers used by plotting (improved) ----------
def _savefig(out_dir: str, filename: str, fig=None, dpi: int = 160) -> str:
    """
    Save current or provided Matplotlib figure to out_dir/filename and return the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _normalize_freq(freq: str) -> str:
    """Forward-compat mapping for pandas frequency aliases."""
    if not isinstance(freq, str):
        return freq
    freq = freq.upper()
    mapping = {
        "M": "ME",   # month-end → MonthEnd
        "MS": "MS",  # month-start → MonthStart
        "Q": "QE",   # quarter-end
        "QS": "QS",  # quarter-start
        "Y": "YE",   # year-end
        "A": "YE",   # alias for annual end
        "AS": "YS",  # year-start
    }
    return mapping.get(freq, freq)

def _nice_time_axis(ax, idx, freq: str, max_ticks: int = 10, rotate: int = 0):
    """
    Clean, compact date ticks by frequency.

    - Accepts DatetimeIndex, PeriodIndex, numpy arrays, or lists of timestamps.
    - Chooses readable tick positions (<= max_ticks).
    - Custom quarter labeling when freq is QS/QE; YYYY-Qk.
    """
    # Normalize/guard inputs
    freq = _normalize_freq(freq)
    # If index is not a DatetimeIndex, try to convert
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(idx)
        except Exception:
            # Fall back to raw tick positions without date formatting
            n = len(idx)
            if n == 0:
                return
            tick_idx = np.arange(n) if n <= max_ticks else np.linspace(0, n - 1, max_ticks, dtype=int)
            ax.set_xticks(tick_idx)
            if rotate:
                for t in ax.get_xticklabels():
                    t.set_rotation(rotate)
            ax.margins(x=0.01)
            return

    n = len(idx)
    if n == 0:
        return
    tick_idx = np.arange(n) if n <= max_ticks else np.linspace(0, n - 1, max_ticks, dtype=int)
    tick_pos = idx[tick_idx]
    ax.set_xticks(tick_pos)

    # Formatter selection
    fmt = None
    if freq == "D":
        fmt = mdates.DateFormatter("%Y-%m-%d")
    elif freq == "W":
        # ISO week label: YYYY-Www
        fmt = mdates.DateFormatter("%G-W%V")
    elif freq in ("MS", "ME"):
        fmt = mdates.DateFormatter("%Y-%m")
    elif freq in ("QS", "QE"):
        # Render custom quarter labels
        labels = [f"{d.year}-Q{((d.month - 1)//3) + 1}" for d in tick_pos]
        ax.set_xticklabels(labels)
    else:
        # Adaptive fallback
        locator = mdates.AutoDateLocator(minticks=4, maxticks=max_ticks)
        fmt = mdates.AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)

    if fmt is not None:
        ax.xaxis.set_major_formatter(fmt)

    if rotate:
        for t in ax.get_xticklabels():
            t.set_rotation(rotate)
    ax.figure.autofmt_xdate()
    ax.margins(x=0.01)

def _has(df: pd.DataFrame, cols) -> bool:
    """Return True if all cols exist in df.columns."""
    return all(c in df.columns for c in cols)
# --- cell ---
# ---------- Robust date parsing (with audit, improved) ----------
_CANDIDATE_FORMATS = [
    # ISO-like
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    # Slash
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    # Dash dayfirst/monthfirst
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y %H:%M:%S",
    # Dots (EU)
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%Y.%m.%d",
    "%Y.%m.%d %H:%M:%S",
]

# Support s / ms / µs epoch lengths
_EPOCH_10_RE = re.compile(r"^\s*\d{10}\s*$")     # seconds
_EPOCH_13_RE = re.compile(r"^\s*\d{13}\s*$")     # milliseconds
_EPOCH_16_RE = re.compile(r"^\s*\d{16}\s*$")     # microseconds

# Precise matcher: 'date', 'datetime', 'timestamp', 'time' as tokens, or *_at suffix (e.g., created_at)
_DATE_NAME_RE = re.compile(r"(?:^|_)(date|datetime|timestamp|time)(?:$|_)", re.I)

def _is_date_like_col(name: str) -> bool:
    n = name.lower()
    if _DATE_NAME_RE.search(n):
        return True
    if n.endswith("_at"):
        return True
    return False


def _guess_epoch_unit(sample: pd.Series, min_match: float = 0.7):
    """
    Return 's', 'ms', or 'us' if >= min_match of non-null values look like UNIX epoch.
    Works for object, integer, or float series.
    """
    s = sample.dropna()
    if s.empty:
        return None
    # Cast to string once
    s = s.astype(str).str.strip()
    n = len(s)
    if n == 0:
        return None
    if (s.str.match(_EPOCH_10_RE).sum() / n) >= min_match:
        return "s"
    if (s.str.match(_EPOCH_13_RE).sum() / n) >= min_match:
        return "ms"
    if (s.str.match(_EPOCH_16_RE).sum() / n) >= min_match:
        return "us"
    return None

def _heuristic_filter_formats(strings: pd.Series) -> list:
    """
    Narrow the candidate formats by simple separators/markers to reduce futile trials.
    """
    sample = strings.dropna().astype(str)
    if sample.empty:
        return _CANDIDATE_FORMATS

    text = " ".join(sample.head(50))  # small peek
    has_t = "T" in text
    has_slash = "/" in text
    has_dash = "-" in text
    has_dot = "." in text
    has_tz = "+" in text or "Z" in text

    candidates = []
    for fmt in _CANDIDATE_FORMATS:
        if has_t and "T" not in fmt:
            continue
        if not has_t and "T" in fmt:
            # Allow non-'T' formats when text doesn't contain 'T'
            continue
        if has_slash and "/" not in fmt:
            continue
        if has_dash and "-" not in fmt and "T" not in fmt:
            continue
        if has_dot and "." not in fmt and "%f" not in fmt:
            # dots often imply European '.' or fractional seconds
            pass  # don't over-filter; allow
        if has_tz and "%z" not in fmt and "T" in fmt:
            # if looks ISO-ish with timezone
            pass  # allow; dateutil will handle fallback anyway
        candidates.append(fmt)

    # Fallback if we filtered too aggressively
    if len(candidates) < 3:
        return _CANDIDATE_FORMATS
    return candidates

def _guess_datetime_format(series: pd.Series, sample_size: int = 200, min_match: float = 0.7):
    """
    Pick the strptime format that parses the highest fraction of a sample.
    Uses heuristic pre-filtering of formats to speed up.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return None
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)

    fmts = _heuristic_filter_formats(s)
    best_fmt, best_score = None, 0.0
    for fmt in fmts:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        score = float(parsed.notna().mean())
        if score > best_score:
            best_fmt, best_score = fmt, score
            if best_score == 1.0:  # perfect match, stop early
                break
    return best_fmt if best_score >= min_match else None

def parse_date_with_audit(
    df: pd.DataFrame,
    col: str,
    *,
    dayfirst_fallback: bool = True
):
    """
    Parse df[col] via epoch → best-format → explicit timezone variants → general fallback.

    Returns
    -------
    parsed_series : pd.Series (dtype datetime64[ns])
    audit_str     : str (how it was parsed)

    Notes
    -----
    - If the series is already datetime64, returns it unchanged.
    - If all values are null, returns all-NaT with a clear audit message.
    - If formats are ambiguous, an optional dayfirst fallback is attempted.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found")

    s = df[col]
    if s.dtype.kind in "Mm":
        return s, f"{col}: already datetime64[ns]"

    if s.isna().all():
        return pd.to_datetime(s, errors="coerce"), f"{col}: all values null → all NaT"

    # 1) Epoch detection
    unit = _guess_epoch_unit(s if s.dtype == "O" else s.astype("object"))
    if unit:
        return pd.to_datetime(s, unit=unit, errors="coerce"), f"{col}: parsed as UNIX epoch ({unit})"

    # 2) Guess explicit strptime format
    fmt = _guess_datetime_format(s)
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce"), f"{col}: parsed with explicit format '{fmt}'"

    # 3) Try explicit timezone-aware ISO if hints exist
    s_str = s.astype(str)
    if s_str.str.contains("Z|\\+\\d{2}:?\\d{2}", regex=True).any():
        for iso_fmt in ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            parsed = pd.to_datetime(s, format=iso_fmt, errors="coerce")
            if parsed.notna().any():
                return parsed, f"{col}: parsed with explicit ISO+tZ format '{iso_fmt}'"

    # 4) General fallback, mixed-format parser.
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    audit = f"{col}: fallback via pd.to_datetime(format='mixed')"

    # 5) Optional dayfirst retry for ambiguous d/m
    if dayfirst_fallback and parsed.isna().mean() > 0 and s_str.str.contains(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", regex=True).any():
        parsed2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        # Prefer the parse that yields fewer NaT
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed, audit = parsed2, f"{col}: fallback via pd.to_datetime (dayfirst=True)"

    return parsed, audit
# --- cell ---
# ---------- GENERIC MISSING-VALUE HANDLER (two parameters, extended) ----------
def handle_missing(col: str, action: str):
    """
    Handle missing values for a single column using a compact action string.

    Parameters
    ----------
    col : str
        Column name (silently no-ops if not present).
    action : str
        One of:
          - 'set_zero' | 'zero'
          - 'set_nan'  | 'nan'
          - 'drop_row' | 'drop'                # drop rows where col is NA
          - 'mean' | 'median' | 'mode'         # global fill
          - 'mean_by:<groupcol>'
          - 'median_by:<groupcol>'
          - 'mode_by:<groupcol>'
          - 'percentile:<q>'                   # e.g., 'percentile:0.5' (=median)
          - 'const:<value>'                    # e.g., 'const:0', 'const:(missing)'
          - 'ffill' | 'bfill'
          - 'interpolate:<method>'             # 'linear' (default), 'time', 'polynomial-2', ...
    Notes
    -----
    - Numeric-only ops (mean/median/percentile/interpolate) are applied only if dtype is numeric.
    - For 'interpolate:time', a DatetimeIndex is expected; falls back to 'linear' if absent.
    - 'mode' works for both numeric/object. If multiple modes, first is used.
    """
    global df
    if col not in df.columns:
        return

    a = action.strip()

    # Helpers
    def _is_num(s): return pd.api.types.is_numeric_dtype(s)
    def _const_value(raw, series):
        # Try to coerce const to column dtype when possible
        raw = raw.strip()
        if raw.lower() in {"nan", "na", "null"}:
            return np.nan
        if _is_num(series):
            try:
                return float(raw)
            except Exception:
                pass
        # For datetimes, try parsing
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(raw, errors="coerce")
            except Exception:
                pass
        return raw  # fallback: string

    # Basic actions
    a_low = a.lower()
    if a_low in {"set_zero", "zero"}:
        df[col] = df[col].fillna(0)
        return
    if a_low in {"set_nan", "nan"}:
        df[col] = df[col].where(~df[col].isna(), np.nan)
        return
    if a_low in {"drop_row", "drop"}:
        df = df[~df[col].isna()]
        return
    if a_low == "mean":
        if _is_num(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        return
    if a_low == "median":
        if _is_num(df[col]):
            df[col] = df[col].fillna(df[col].median())
        return
    if a_low == "mode":
        if df[col].isna().any():
            m = df[col].mode(dropna=True)
            if not m.empty:
                df[col] = df[col].fillna(m.iloc[0])
        return
    if a_low in {"ffill", "bfill"}:
        df[col] = df[col].fillna(method=a_low)
        return

    # Parameterized: const:<value>
    if a_low.startswith("const:"):
        raw = a.split(":", 1)[1]
        val = _const_value(raw, df[col])
        df[col] = df[col].fillna(val)
        return

    # Parameterized: percentile:<q>
    if a_low.startswith("percentile:"):
        if _is_num(df[col]):
            try:
                q = float(a.split(":", 1)[1])
                q = min(max(q, 0.0), 1.0)
                val = df[col].quantile(q)
                df[col] = df[col].fillna(val)
            except Exception:
                pass
        return

    # Parameterized: mean_by:/median_by:/mode_by:<groupcol>
    if any(a_low.startswith(prefix) for prefix in ("mean_by:", "median_by:", "mode_by:")):
        try:
            method, grp = a_low.split(":", 1)
            grp = grp.strip()
            if grp in df.columns and df[col].isna().any():
                if method == "mean_by" and _is_num(df[col]):
                    fill_vals = df.groupby(grp)[col].transform("mean")
                    df[col] = df[col].fillna(fill_vals)
                elif method == "median_by" and _is_num(df[col]):
                    fill_vals = df.groupby(grp)[col].transform("median")
                    df[col] = df[col].fillna(fill_vals)
                elif method == "mode_by":
                    # Works for numeric or object
                    def _grp_mode(s):
                        m = s.mode(dropna=True)
                        return m.iloc[0] if not m.empty else np.nan
                    fill_vals = df.groupby(grp)[col].transform(_grp_mode)
                    df[col] = df[col].fillna(fill_vals)
        except Exception:
            pass
        return

    # Parameterized: interpolate:<method>
    if a_low.startswith("interpolate:"):
        method = a.split(":", 1)[1].strip()
        if not method:
            method = "linear"
        if _is_num(df[col]):
            try:
                if method == "time" and not isinstance(df.index, pd.DatetimeIndex):
                    # Fallback if no DatetimeIndex
                    method = "linear"
                df[col] = df[col].interpolate(method=method, limit_direction="both")
            except Exception:
                # Silent fallback: leave as is
                pass
        return

    # Unknown action -> no-op
    return

# --- cell ---
# ---------- LOAD ----------
df = pd.read_csv(RAW_PATH)

# ---------- STANDARDIZE STRINGS ----------
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()

# ---------- DATETIME PARSING (AUTO via helper) ----------
parsed_report = []  # keep an audit trail
for c in df.columns:
    # skip if already datetime
    if df[c].dtype.kind in "Mm":
        continue
    # only parse true date-like columns (avoids matching 'category')
    if _is_date_like_col(c):
        parsed, audit = parse_date_with_audit(df, c)  # uses epoch→format→mixed fallback
        df[c] = parsed
        parsed_report.append((c, audit))


# ---------- NUMERIC COERCION ----------
for c in df.columns:
    if df[c].dtype == "object":
        try:
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().mean() >= 0.7:
                df[c] = as_num
        except Exception:
            pass

# ---------- NORMALIZE MISSING VALUES ----------
df = df.replace(
    to_replace=["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "-"],
    value=np.nan
)

# ---------- DEDUP ----------
df = df.drop_duplicates()

# ---------- DOMAIN RULES FOR E-COMMERCE ----------
# 1) Deterministic calculations for qty / unit_price / revenue when possible.
cols = {c.lower(): c for c in df.columns}  # case-insensitive map
qty_col     = cols.get("qty")
price_col   = cols.get("unit_price")
revenue_col = cols.get("revenue")
product_col = cols.get("product")

if qty_col and price_col and revenue_col:
    # qty = revenue / unit_price
    mask = df[qty_col].isna() & df[revenue_col].notna() & df[price_col].notna() & (df[price_col] != 0)
    df.loc[mask, qty_col] = df.loc[mask, revenue_col] / df.loc[mask, price_col]

    # unit_price = revenue / qty
    mask = df[price_col].isna() & df[revenue_col].notna() & df[qty_col].notna() & (df[qty_col] != 0)
    df.loc[mask, price_col] = df.loc[mask, revenue_col] / df.loc[mask, qty_col]

    # revenue = qty * unit_price
    mask = df[revenue_col].isna() & df[qty_col].notna() & df[price_col].notna()
    df.loc[mask, revenue_col] = df.loc[mask, qty_col] * df.loc[mask, price_col]

# 1b) If unit_price still missing but product is known, use median-by-product (uses the generic handler)
if price_col and product_col and df[price_col].isna().any():
    handle_missing(price_col, f"median_by:{product_col}")

# 2) For categorical IDs/names: keep as NaN (explicitly set to NaN; do NOT fill with mode)
for keep_nan_col in ["country", "product", "category"]:
    if keep_nan_col in df.columns:
        handle_missing(keep_nan_col, "set_nan")

# ---------- LIGHT IMPUTATIONS (preserve intent, use generic handler) ----------
# Numeric: fill median for numeric columns EXCEPT qty/unit_price/revenue (leave NaN if not calculable)
numeric_exclude = {x for x in [qty_col, price_col, revenue_col] if x}
for c in df.select_dtypes(include=[np.number]).columns:
    if c not in numeric_exclude and df[c].isnull().any():
        handle_missing(c, "median")

# Object: fill mode for object columns EXCEPT country/product/category (these must remain NaN if missing)
object_exclude = {"country", "product", "category"}
for c in df.select_dtypes(include=["object"]).columns:
    if c not in object_exclude and df[c].isnull().any():
        handle_missing(c, "mode")

# Datetime: forward-fill is fine for temporal continuity
for c in df.select_dtypes(include=["datetime64[ns]"]).columns:
    if df[c].isnull().any():
        handle_missing(c, "ffill")

# ---------- SAVE ----------
# Write visible NaNs so Excel doesn’t show blanks
df.to_csv(CLEAN_PATH, index=False, na_rep="NaN")

# ---------- AUDIT LOG ----------
print("Saved cleaned dataset to", CLEAN_PATH)
if parsed_report:
    print("\n[Datetime parsing audit]")
    for col, how in parsed_report:
        print(f" - {col}: {how}")
# --- cell ---
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== Lightweight visual theme helpers (non-breaking) =====================

# Subtle, readable color cycle (tab10) expanded for bars/slices
_PALETTE = plt.cm.tab20.colors  # tuple of RGBA colors

mpl.rcParams.update({
    "figure.autolayout": False,  # we'll still call tight_layout() before save
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.prop_cycle": mpl.cycler(color=[
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
    ]),
    "grid.linestyle": "-",
    "grid.alpha": 0.3,
})


def _apply_theme(ax):
    """Apply a clean, modern look without altering chart logic."""
    ax.set_facecolor("#f9fafb")
    ax.grid(True)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.figure.set_facecolor("#ffffff")


def _colors(n):
    """Return n distinct colors from the palette."""
    base = list(_PALETTE)
    if n <= len(base):
        return base[:n]
    # repeat if more colors are needed
    reps = (n + len(base) - 1) // len(base)
    return (base * reps)[:n]


# ===================== E-commerce Charts — Callable Functions =====================
# Requirements: pandas, numpy, matplotlib
import matplotlib.dates as mdates

# ---------- I/O helper ----------
def _savefig(out_dir: str, filename: str, fig=None, dpi: int = 160):
    os.makedirs(out_dir, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Date-axis helper ----------
def _nice_time_axis(ax, idx, freq: str, max_ticks: int = 10, rotate: int = 0):
    n = len(idx)
    if n == 0:
        return
    tick_idx = np.arange(n) if n <= max_ticks else np.linspace(0, n - 1, max_ticks, dtype=int)
    ax.set_xticks(idx[tick_idx])

    if freq == "D":
        fmt = mdates.DateFormatter("%Y-%m-%d")
    elif freq == "W":
        fmt = mdates.DateFormatter("%Y-W%W")
    elif freq in ("MS", "ME"):
        fmt = mdates.DateFormatter("%Y-%m")
    else:
        fmt = mdates.AutoDateFormatter(mdates.AutoDateLocator(minticks=4, maxticks=max_ticks))
    ax.xaxis.set_major_formatter(fmt)

    if rotate:
        for t in ax.get_xticklabels():
            t.set_rotation(rotate)
    ax.figure.autofmt_xdate()
    ax.margins(x=0.01)

# ---------- 1) Quantity per period (replaces old histogram) ----------
def plot_qty_by_period(
    df: pd.DataFrame,
    date_col: str = "order_date",
    qty_col: str = "qty",
    freq: str = "W",                                # "D","W","MS","ME"
    out_dir: str = "charts",
    outfile: str = "1_qty_by_week.png",
    rotate_xticks: int = 45,
    max_ticks: int = 10
):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    qty = pd.to_numeric(df[qty_col], errors="coerce")
    ok = dates.notna() & qty.notna()
    ts = (pd.DataFrame({date_col: dates[ok], qty_col: qty[ok]})
          .set_index(date_col).resample(freq)[qty_col].sum())
    fig, ax = plt.subplots(figsize=(12,5))
    width = 6 if freq == "W" else 20 if freq in ("MS","ME") else 0.9
    colors = _colors(len(ts))
    ax.bar(ts.index, ts.values, width=width, color=colors, edgecolor="#ffffff", linewidth=0.6)
    label = "Day" if freq=="D" else "Week" if freq=="W" else "Month"
    ax.set_title(f"Quantity per {label}"); ax.set_xlabel("Date"); ax.set_ylabel("Quantity")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 2) Revenue by period (line/bar) ----------
def plot_revenue_by_period(
    df: pd.DataFrame,
    date_col: str = "order_date",
    revenue_col: str = "revenue",
    freq: str = "ME",                                 # "D","W","MS","ME"
    kind: str = "line",                              # "line" or "bar"
    out_dir: str = "charts",
    outfile: str = "2_monthly_revenue.png",
    rotate_xticks: int = 0,
    max_ticks: int = 10
):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    rev = pd.to_numeric(df[revenue_col], errors="coerce")
    ok = dates.notna() & rev.notna()
    ts = (pd.DataFrame({date_col: dates[ok], revenue_col: rev[ok]})
          .set_index(date_col).resample(freq)[revenue_col].sum())
    fig, ax = plt.subplots(figsize=(12,5))
    if kind == "bar":
        width = 6 if freq == "W" else 20 if freq in ("MS","ME") else 0.9
        colors = _colors(len(ts))
        ax.bar(ts.index, ts.values, width=width, color=colors, edgecolor="#ffffff", linewidth=0.6)
    else:
        ax.plot(ts.index, ts.values, linewidth=2.2, marker="o", markersize=4)
    label = "Day" if freq=="D" else "Week" if freq=="W" else "Month"
    ax.set_title(f"Revenue per {label}"); ax.set_xlabel("Date"); ax.set_ylabel("Revenue")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 3) Top Countries by Revenue (bar) ----------
def plot_top_countries_by_revenue(
    df: pd.DataFrame,
    country_col: str = "country",
    revenue_col: str = "revenue",
    top_n: int = 5,
    out_dir: str = "charts",
    outfile: str = "3_revenue_by_country.png"
):
    s = (df.groupby(country_col)[revenue_col].sum()
           .sort_values(ascending=False).head(top_n))
    fig, ax = plt.subplots(figsize=(10,5))
    colors = _colors(len(s))
    s.plot(kind="bar", ax=ax, color=colors, edgecolor="#ffffff")
    ax.set_title("Top Countries by Revenue"); ax.set_xlabel("Country"); ax.set_ylabel("Revenue")
    for p in ax.patches:
        p.set_linewidth(0.6)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 4) Unit Price vs Revenue (log-Y) — replaces bad order_id scatter ----------
def plot_unitprice_vs_revenue(
    df: pd.DataFrame,
    price_col: str = "unit_price",
    revenue_col: str = "revenue",
    out_dir: str = "charts",
    outfile: str = "4_scatter.png"
):
    x = pd.to_numeric(df[price_col], errors="coerce")
    y = pd.to_numeric(df[revenue_col], errors="coerce")
    ok = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(x[ok], y[ok], alpha=0.7, s=28, linewidths=0.5, edgecolors="#ffffff")
    ax.set_yscale("log")
    ax.set_title("Unit Price vs Revenue (log scale)")
    ax.set_xlabel("Unit Price"); ax.set_ylabel("Revenue (log)")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 5) Revenue per Order by Product (Top 10) — Boxplot ----------
def plot_revenue_by_product_boxplot(
    df: pd.DataFrame,
    product_col: str = "product",
    revenue_col: str = "revenue",
    top_n: int = 10,
    out_dir: str = "charts",
    outfile: str = "5_box_by_cat.png"
):
    totals = (df.groupby(product_col)[revenue_col].sum()
                .sort_values(ascending=False).head(top_n))
    top_idx = totals.index
    sub = df[df[product_col].isin(top_idx)]
    fig, ax = plt.subplots(figsize=(12,6))
    bp = sub.boxplot(column=revenue_col, by=product_col, rot=45, ax=ax, patch_artist=True)
    # Color boxes for better readability
    for i, box in enumerate(ax.artists):
        box.set_facecolor(_colors(len(top_idx))[i % len(top_idx)])
        box.set_edgecolor("#ffffff"); box.set_linewidth(0.6)
    ax.set_title("Revenue per Order by Product (Top 10)")
    plt.suptitle("")
    ax.set_xlabel("Product"); ax.set_ylabel("Revenue per Order")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 6) Revenue Share by Category (pie) ----------
def plot_category_share_pie(
    df: pd.DataFrame,
    category_col: str = "category",
    revenue_col: str = "revenue",
    out_dir: str = "charts",
    outfile: str = "6_category_share.png"
):
    s = df.groupby(category_col)[revenue_col].sum()
    fig, ax = plt.subplots(figsize=(7,7))
    colors = _colors(len(s))
    ax.pie(s.values, labels=s.index, autopct="%1.0f%%", startangle=90,
           colors=colors, wedgeprops={"edgecolor": "#ffffff", "linewidth": 0.8},
           textprops={"fontsize": 10})
    ax.set_title("Revenue Share by Category")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 7) Unit Price vs Quantity (scatter) ----------
def plot_price_vs_qty_scatter(
    df: pd.DataFrame,
    price_col: str = "unit_price",
    qty_col: str = "qty",
    out_dir: str = "charts",
    outfile: str = "7_price_vs_qty.png"
):
    x = pd.to_numeric(df[price_col], errors="coerce")
    y = pd.to_numeric(df[qty_col], errors="coerce")
    ok = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(x[ok], y[ok], alpha=0.7, marker='x', s=28)
    ax.set_title("Unit Price vs Quantity")
    ax.set_xlabel("Unit Price"); ax.set_ylabel("Quantity")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 8) Daily Orders (count) time series ----------
def plot_daily_orders(
    df: pd.DataFrame,
    date_col: str = "order_date",
    id_col: str = "order_id",           # counts distinct orders per day
    out_dir: str = "charts",
    outfile: str = "8_timeseries.png",
    rotate_xticks: int = 0,
    max_ticks: int = 12
):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    ids = df[id_col]
    ok = dates.notna() & ids.notna()
    # count distinct orders per day
    ts = (pd.DataFrame({date_col: dates[ok], id_col: ids[ok]})
          .groupby(pd.Grouper(key=date_col, freq="D"))[id_col]
          .nunique())
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(ts.index, ts.values, linewidth=2.0, marker="o", markersize=4)
    ax.set_title("Daily Order Count"); ax.set_xlabel("Date"); ax.set_ylabel("Orders")
    _nice_time_axis(ax, ts.index, freq="D", max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 9) Correlation Heatmap (exclude IDs/high-cardinality) ----------
def plot_corr_heatmap_wo_ids(
    df: pd.DataFrame,
    out_dir: str = "charts",
    outfile: str = "9_corr_heatmap.png",
    id_suffixes: tuple = ("id",),
    uniq_ratio_cut: float = 0.9
):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = []
    for c in num_cols:
        cl = c.lower()
        if any(cl.endswith(suf) for suf in id_suffixes):
            drop_cols.append(c); continue
        unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > uniq_ratio_cut:
            drop_cols.append(c)
    use_cols = [c for c in num_cols if c not in drop_cols]
    if len(use_cols) < 2:
        return  # not enough numeric cols after filtering
    corr = df[use_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7,6))
    img = ax.imshow(corr.values, interpolation="nearest", cmap="viridis")
    ax.set_title("Correlation Heatmap (IDs excluded)")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=12)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ---------- 10) Revenue distribution (histogram) ----------
def plot_revenue_hist(
    df: pd.DataFrame,
    revenue_col: str = "revenue",
    out_dir: str = "charts",
    outfile: str = "10_revenue_hist.png",
    bins: int = 40
):
    rev = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(rev, bins=bins, edgecolor="#ffffff", linewidth=0.6, alpha=0.9)
    ax.set_title("Revenue Distribution per Order")
    ax.set_xlabel("Revenue"); ax.set_ylabel("Count")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# ===================== Example usage =====================
# df = pd.read_csv("ecommerce_clean.csv")
# plot_qty_by_period(df, freq="W", outfile="1_qty_by_week.png")
# plot_revenue_by_period(df, freq="ME", kind="line", outfile="1_monthly_revenue.png")
# plot_top_countries_by_revenue(df, top_n=5)
# plot_unitprice_vs_revenue(df)
# plot_revenue_by_product_boxplot(df)
# plot_category_share_pie(df)
# plot_price_vs_qty_scatter(df)
# plot_daily_orders(df)
# plot_corr_heatmap_wo_ids(df)
# plot_revenue_hist(df)

# --- cell ---
# ---------------- Run each plot once with column checks ----------------
def _has(df, cols): 
    return all(c in df.columns for c in cols)

# 1) Quantity per period (weekly bar)
if _has(df, ["order_date", "qty"]):
    plot_qty_by_period(
        df, date_col="order_date", qty_col="qty",
        freq="W", outfile="1_qty_by_week.png", rotate_xticks=45, max_ticks=8
    )

# 2) Revenue by period (monthly line)
if _has(df, ["order_date", "revenue"]):
    plot_revenue_by_period(
        df, date_col="order_date", revenue_col="revenue",
        freq="ME", kind="line", outfile="2_monthly_revenue.png", rotate_xticks=0, max_ticks=10
    )

# 3) Top countries by revenue (bar)
if _has(df, ["country", "revenue"]):
    plot_top_countries_by_revenue(
        df, country_col="country", revenue_col="revenue",
        top_n=5, outfile="3_revenue_by_country.png"
    )

# 4) Unit price vs revenue (log-Y) — replaces old order_id scatter
if _has(df, ["unit_price", "revenue"]):
    plot_unitprice_vs_revenue(
        df, price_col="unit_price", revenue_col="revenue",
        outfile="4_scatter.png"
    )

# 5) Revenue per order by product (Top 10) — boxplot
if _has(df, ["product", "revenue"]):
    plot_revenue_by_product_boxplot(
        df, product_col="product", revenue_col="revenue",
        top_n=10, outfile="5_box_by_cat.png"
    )

# 6) Revenue share by category (pie)
if _has(df, ["category", "revenue"]):
    plot_category_share_pie(
        df, category_col="category", revenue_col="revenue",
        outfile="6_category_share.png"
    )

# 7) Unit price vs quantity (scatter)
if _has(df, ["unit_price", "qty"]):
    plot_price_vs_qty_scatter(
        df, price_col="unit_price", qty_col="qty",
        outfile="7_price_vs_qty.png"
    )

# 8) Daily orders (distinct order_id per day)
if _has(df, ["order_date", "order_id"]):
    plot_daily_orders(
        df, date_col="order_date", id_col="order_id",
        outfile="8_timeseries.png", rotate_xticks=0, max_ticks=12
    )

# 9) Correlation heatmap (IDs excluded) — relies on numeric columns; function self-guards
plot_corr_heatmap_wo_ids(
    df, outfile="9_corr_heatmap.png", id_suffixes=("id",), uniq_ratio_cut=0.9
)

# 10) Revenue distribution (histogram)
if _has(df, ["revenue"]):
    plot_revenue_hist(
        df, revenue_col="revenue",
        outfile="10_revenue_hist.png", bins=40
    )
