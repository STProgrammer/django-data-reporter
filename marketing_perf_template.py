# --- cell ---
# ===================== Marketing Performance — Cleaner + Plots (Robust Dates) =====================
# Raw columns expected (case-insensitive; spaces allowed and normalized):
#   date, channel, spend, clicks, conversions, revenue
#
# Outputs:
#   - marketing_clean.csv  (adds: cpc, cvr, roas)
#   - charts/1_spend_by_channel.png
#   - charts/2_roas_by_channel.png
#   - charts/3_cvr_boxplot.png
#   - charts/4_spend_timeseries.png
#   - charts/5_revenue_vs_spend.png
#
# Requirements: Python 3.9+, pandas>=2.2, matplotlib>=3.9, numpy

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- I/O paths ----------
RAW_PATH   = "marketing_raw.csv"
CLEAN_PATH = "marketing_clean.csv"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)
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

    # 4) General fallback (dateutil)
    parsed = pd.to_datetime(s, errors="coerce")
    audit = f"{col}: fallback via pd.to_datetime (dateutil parser)"

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
# ===================== 1) LOAD + CLEAN =====================
df_raw = pd.read_csv(RAW_PATH)

# Normalize column names (lower + underscores) to be tolerant to raw headers
df_raw.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df_raw.columns]

required = ["date", "channel", "spend", "clicks", "conversions", "revenue"]
if not _has(df_raw, required):
    missing = sorted(list(set(required) - set(df_raw.columns)))
    raise ValueError(f"Missing required columns in RAW: {missing}")

df = df_raw.copy()

# Trim strings in object cols
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()

# Normalize obvious placeholders → NaN up-front
df = df.replace(
    to_replace=["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "-"],
    value=np.nan
)

# Canonicalize channel names
df["channel"] = df["channel"].astype("string").str.strip().str.title()

# Robust date parsing (with audit) via helper (avoids deprecated inference path)
parsed_date, date_audit = parse_date_with_audit(df, "date")
df["date"] = parsed_date
print("[Date parsing audit]")
print(" -", date_audit)

# Numeric coercions (explicit; no deprecated flags)
for c in ["spend", "clicks", "conversions", "revenue"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Enforce intended missing-policy via generic handler
# (Keep channel missing as real NaN; do not mode-fill.)
handle_missing("channel", "set_nan")

# Drop rows lacking key identifiers (date or channel)
df = df[df["date"].notna() & df["channel"].notna()]

# Derived metrics (NaN if denominator <= 0)
df["cpc"]  = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], np.nan)
df["cvr"]  = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], np.nan)
df["roas"] = np.where(df["spend"]  > 0, df["revenue"] / df["spend"],  np.nan)

# De-duplicate
df = df.drop_duplicates()

# Save clean file (write visible NaNs so Excel doesn’t show blanks)
df.to_csv(CLEAN_PATH, index=False, na_rep="NaN")
print(f"Saved cleaned dataset to {CLEAN_PATH}")
# --- cell ---
# Lightweight theme helpers (do not change your existing helpers like save_chart / require_cols / nice_time_axis)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Cohesive palette & rcParams that won't break your logic
mpl.rcParams.update({
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "grid.alpha": 0.3,
})

_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
]

def _colors(n: int):
    base = list(_PALETTE)
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def _apply_theme(ax):
    ax.set_facecolor("#f9fafb")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ===================== 2) PLOTTING FUNCTIONS =====================

def plot_total_spend_by_channel(df: pd.DataFrame, outfile: str = "1_spend_by_channel.png"):
    """Bar: total spend by channel."""
    if not require_cols(df, ["channel", "spend"]):
        return
    s = (df.groupby("channel", dropna=False)["spend"].sum().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    colors = _colors(len(s))
    s.plot(kind="bar", ax=ax, color=colors, edgecolor="#ffffff")
    ax.set_title("Total Spend by Channel")
    ax.set_xlabel("Channel"); ax.set_ylabel("Spend")
    _apply_theme(ax)
    save_chart(fig, outfile)


def plot_avg_roas_by_channel(df: pd.DataFrame, outfile: str = "2_roas_by_channel.png"):
    """Bar: mean ROAS by channel."""
    if not require_cols(df, ["channel", "roas"]):
        return
    s = (df.dropna(subset=["roas"])\
           .groupby("channel", dropna=False)["roas"].mean()\
           .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    colors = _colors(len(s))
    s.plot(kind="bar", ax=ax, color=colors, edgecolor="#ffffff")
    ax.set_title("Average ROAS by Channel")
    ax.set_xlabel("Channel"); ax.set_ylabel("ROAS")
    _apply_theme(ax)
    save_chart(fig, outfile)


def plot_cvr_boxplot(df: pd.DataFrame, outfile: str = "3_cvr_boxplot.png"):
    """Boxplot: CVR distribution by channel (uses Matplotlib 3.9+ tick_labels)."""
    if not require_cols(df, ["channel", "cvr"]):
        return
    d = df.dropna(subset=["cvr"])
    if d.empty:
        return
    order = d.groupby("channel")["cvr"].median().sort_values(ascending=False).index
    data = [d.loc[d["channel"] == ch, "cvr"].to_numpy(dtype=float) for ch in order]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=order)  # no deprecated 'labels'
    # Color each box subtly
    cols = _colors(len(order))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=cols[i], edgecolor="#ffffff", linewidth=0.8, alpha=0.95)
    for item in bp["whiskers"] + bp["caps"]:
        item.set(color="#666666", linewidth=0.8)
    for med in bp["medians"]:
        med.set(color="#1f1f1f", linewidth=1.4)
    ax.set_title("CVR by Channel")
    ax.set_xlabel("Channel"); ax.set_ylabel("CVR")
    _apply_theme(ax)
    save_chart(fig, outfile)


def plot_spend_timeseries_by_channel(df: pd.DataFrame, outfile: str = "4_spend_timeseries.png",
                                     last_n_days: int | None = 60):
    """Line: spend per day by channel (optionally last N days)."""
    if not require_cols(df, ["date", "channel", "spend"]):
        return
    d = df.copy()
    if last_n_days is not None:
        cutoff = d["date"].max() - pd.Timedelta(days=last_n_days - 1)
        d = d[d["date"] >= cutoff]
    ts = (d.groupby(["date", "channel"], as_index=False)["spend"].sum()
            .pivot(index="date", columns="channel", values="spend")
            .sort_index())
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cols = _colors(len(ts.columns))
    for i, ch in enumerate(ts.columns):
        ax.plot(ts.index, ts[ch], label=ch, linewidth=2.0, marker="o", markersize=3.2, color=cols[i])
    ax.set_title(f"Spend by Channel{' (Last ' + str(last_n_days) + ' Days)' if last_n_days else ''}")
    ax.set_xlabel("Date"); ax.set_ylabel("Spend")
    ax.legend(ncol=3, frameon=False)
    nice_time_axis(ax, ts.index.to_numpy(), max_ticks=9, rotate=0)
    _apply_theme(ax)
    save_chart(fig, outfile)


def plot_revenue_vs_spend_scatter(df: pd.DataFrame, outfile: str = "5_revenue_vs_spend.png"):
    """Scatter: total revenue vs total spend by channel with annotations."""
    if not require_cols(df, ["channel", "spend", "revenue"]):
        return
    agg = df.groupby("channel", dropna=False).agg(spend=("spend", "sum"),
                                                  revenue=("revenue", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agg["spend"], agg["revenue"], marker="o", s=80, alpha=0.9,
               edgecolors="#ffffff", linewidths=0.6, color="#4e79a7")
    for _, r in agg.iterrows():
        ax.annotate(str(r["channel"]), (r["spend"], r["revenue"]),
                    xytext=(6, 4), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.75))
    ax.set_title("Revenue vs Spend by Channel")
    ax.set_xlabel("Spend"); ax.set_ylabel("Revenue")
    _apply_theme(ax)
    save_chart(fig, outfile)
# --- cell ---
# ===================== 3) RUN — with guards =====================
if require_cols(df, ["channel", "spend"]):
    plot_total_spend_by_channel(df, outfile="1_spend_by_channel.png")

if require_cols(df, ["channel", "roas"]):
    plot_avg_roas_by_channel(df, outfile="2_roas_by_channel.png")

if require_cols(df, ["channel", "cvr"]):
    plot_cvr_boxplot(df, outfile="3_cvr_boxplot.png")

if require_cols(df, ["date", "channel", "spend"]):
    plot_spend_timeseries_by_channel(df, outfile="4_spend_timeseries.png", last_n_days=60)

if require_cols(df, ["channel", "spend", "revenue"]):
    plot_revenue_vs_spend_scatter(df, outfile="5_revenue_vs_spend.png")
