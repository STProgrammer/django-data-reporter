# --- cell ---
# ===================== SaaS Churn — Load + Clean (Robust Dates) =====================
# Raw columns expected (case-sensitive):
# user_id, signup_date, plan, country, monthly_churn_prob, churn_month, arpu
# (raw has NO 'churned'; we derive it)
#
# Outputs:
# - saas_clean.csv
# - charts/*.png (generated elsewhere)

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- Paths ----------
RAW_PATH   = "saas_raw.csv"
CLEAN_PATH = "saas_clean.csv"
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
# ---------- Load ----------
df_raw = pd.read_csv(RAW_PATH)

# ---------- Cleaning (safe & minimal) ----------
# 0) Ensure required column(s)
if not _has(df_raw, ["signup_date"]):
    missing = [c for c in ["signup_date"] if c not in df_raw.columns]
    raise ValueError(f"Missing required columns: {missing}")

df = df_raw.copy()

# 1) Trim strings early
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()

# 1b) Normalize obvious placeholders → NaN (consistent policy)
df = df.replace(
    to_replace=["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "-"],
    value=np.nan
)

# 2) Parse signup_date robustly + audit (helper avoids deprecated inference)
parsed, audit = parse_date_with_audit(df, "signup_date")
df["signup_date"] = parsed
print("[Date parsing audit]")
print(" -", audit)

# 3) Canonicalize categoricals (title-case), but keep real NaNs
for c in ["plan", "country"]:
    if c in df.columns:
        df[c] = df[c].astype("string").str.title()
        handle_missing(c, "set_nan")  # make policy explicit

# 4) Numeric coercions (explicit; forward-compatible)
for c in ["monthly_churn_prob", "churn_month", "arpu"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 5) Derive churned from churn_month
#    Rule: churned = 1 if churn_month > 0, else 0 (0 means not churned yet).
if "churn_month" in df.columns:
    cm = pd.to_numeric(df["churn_month"], errors="coerce")
    df["churned"] = (cm.fillna(0) > 0).astype(int)
else:
    if "churned" in df.columns:
        del df["churned"]

# 6) Final typing / validations
if "churn_month" in df.columns:
    # Keep NaNs as zero *by design* here, then clamp and cast
    df["churn_month"] = cm.fillna(0).clip(lower=0).round().astype(int)

if "monthly_churn_prob" in df.columns:
    df["monthly_churn_prob"] = df["monthly_churn_prob"].clip(lower=0)

if "arpu" in df.columns:
    df["arpu"] = df["arpu"].clip(lower=0)

# 7) Drop duplicates; light numeric impute with generic handler (exclude churn_month)
df = df.drop_duplicates()
numeric_exclude = {"churn_month"}  # policy: don’t median-impute churn_month
for c in df.select_dtypes(include=[np.number]).columns:
    if c not in numeric_exclude and df[c].isna().any():
        handle_missing(c, "median")

# ---------- Save cleaned ----------
df.to_csv(CLEAN_PATH, index=False, na_rep="NaN")
print(f"Saved cleaned dataset to {CLEAN_PATH}")
# --- cell ---
# Visual polish helpers (non-breaking): keep your existing _has/_normalize_freq/_nice_time_axis/_savefig/CHARTS_DIR as-is
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "grid.alpha": 0.32,
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


# ===================== Callable Plot Functions (10) =====================

# 1) Signups per period (bar)
def plot_signups_by_period(
    df: pd.DataFrame,
    date_col: str = "signup_date",
    freq: str = "W",                 # "D","W","MS","ME"
    out_dir: str = CHARTS_DIR,
    outfile: str = "1_signups_by_week.png",
    rotate_xticks: int = 45,
    max_ticks: int = 10
):
    if not _has(df, [date_col]):
        return
    freq = _normalize_freq(freq)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    ok = dates.notna()
    ts = (pd.DataFrame({date_col: dates[ok]})
          .set_index(date_col).resample(freq).size())
    fig, ax = plt.subplots()
    width = 6 if freq == "W" else 20 if freq in ("MS","ME") else 0.9
    ax.bar(ts.index, ts.values, width=width, color=_colors(len(ts)), edgecolor="#ffffff", linewidth=0.6)
    label = "Day" if freq=="D" else "Week" if freq=="W" else "Month"
    ax.set_title(f"New Signups per {label}"); ax.set_xlabel("Date"); ax.set_ylabel("Signups")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 2) Plan mix (bar)
def plot_plan_mix(
    df: pd.DataFrame,
    plan_col: str = "plan",
    out_dir: str = CHARTS_DIR,
    outfile: str = "2_plan_mix.png"
):
    if not _has(df, [plan_col]):
        return
    s = df[plan_col].value_counts()
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    ax.set_title("Plan Mix"); ax.set_xlabel("Plan"); ax.set_ylabel("Users")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 3) Churn rate by plan (bar) — robust
def plot_churn_rate_by_plan(
    df: pd.DataFrame,
    plan_col: str = "plan",
    churned_col: str = "churned",
    out_dir: str = "charts",
    outfile: str = "3_churn_rate_by_plan.png",
    min_signups: int = 1  # set >1 to hide tiny cohorts
):
    if plan_col not in df.columns or churned_col not in df.columns:
        return False

    # Clean/standardize
    plans = df[plan_col].astype(str).str.strip()
    churned = pd.to_numeric(df[churned_col], errors="coerce")

    # Force binary {0,1}
    churned = churned.fillna(0).clip(0, 1).round().astype(int)

    mask = plans.notna()
    g = pd.DataFrame({plan_col: plans[mask], churned_col: churned[mask]})
    if g.empty: 
        return False

    agg = g.groupby(plan_col, dropna=True).agg(
        churn_rate=(churned_col, "mean"),
        signups=(churned_col, "size")
    )
    agg = agg[agg["signups"] >= min_signups].sort_values("churn_rate", ascending=False)
    if agg.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    agg["churn_rate"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    ax.set_title("Churn Rate by Plan")
    ax.set_xlabel("Plan"); ax.set_ylabel("Churn Rate")
    # annotate counts
    for i, (idx, row) in enumerate(agg.iterrows()):
        ax.text(i, row["churn_rate"] + 0.01, f"n={row['signups']}", ha="center", va="bottom", fontsize=9)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)
    return True


# 4) ARPU by plan (bar)
def plot_arpu_by_plan(
    df: pd.DataFrame,
    plan_col: str = "plan",
    arpu_col: str = "arpu",
    out_dir: str = CHARTS_DIR,
    outfile: str = "4_arpu_by_plan.png"
):
    if not _has(df, [plan_col, arpu_col]):
        return
    s = df.groupby(plan_col)[arpu_col].mean()
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    ax.set_title("ARPU by Plan"); ax.set_xlabel("Plan"); ax.set_ylabel("ARPU ($)")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 5) Retention curves by plan (months since signup; theoretical from churn prob or empirical from churn_month)
def plot_retention_curves(
    df: pd.DataFrame,
    plan_col: str = "plan",
    churn_prob_col: str = "monthly_churn_prob",
    churn_month_col: str = "churn_month",
    months: int = 12,
    out_dir: str = CHARTS_DIR,
    outfile: str = "5_retention_curves.png"
):
    if not _has(df, [plan_col]):
        return
    fig, ax = plt.subplots()
    plotted = False

    # Prefer empirical survival if churn_month present
    if churn_month_col in df.columns:
        for i, (plan, g) in enumerate(df.groupby(plan_col)):
            cm = g[churn_month_col].fillna(0).clip(lower=0).astype(int).to_numpy()
            surv = []
            for m in range(0, months+1):
                # survived beyond m if churn_month==0 (not churned in window) or > m
                survived = ((cm == 0) | (cm > m)).mean()
                surv.append(survived)
            ax.plot(range(0, months+1), surv, label=str(plan), linewidth=2.0, marker="o", markersize=3.2,
                    color=_colors(1)[0] if len(df[plan_col].unique())==1 else _colors(len(df[plan_col].unique()))[i])
            plotted = True

    # Fallback: theoretical geometric survival from monthly churn prob
    elif churn_prob_col in df.columns:
        for i, (plan, g) in enumerate(df.groupby(plan_col)):
            p = g[churn_prob_col].dropna().median()
            surv = [(1 - p) ** m for m in range(0, months+1)]
            ax.plot(range(0, months+1), surv, label=str(plan), linewidth=2.0, marker="o", markersize=3.2,
                    color=_colors(len(df[plan_col].nunique()))[i])
            plotted = True

    if not plotted:
        plt.close(fig); return

    ax.set_title("Retention Curves by Plan (0–12 months)")
    ax.set_xlabel("Months since Signup"); ax.set_ylabel("Retention")
    ax.legend(frameon=False)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 6) MRR timeline (month-end) — sum ARPU across active users per calendar month
def plot_mrr_timeline(
    df: pd.DataFrame,
    user_col: str = "user_id",
    signup_col: str = "signup_date",
    churn_month_col: str = "churn_month",
    arpu_col: str = "arpu",
    out_dir: str = CHARTS_DIR,
    outfile: str = "6_mrr_timeline.png",
    freq: str = "ME",
    rotate_xticks: int = 45,
    max_ticks: int = 12
):
    if not _has(df, [user_col, signup_col, arpu_col]):
        return
    freq = _normalize_freq(freq)

    # bounds
    sdates = pd.to_datetime(df[signup_col], errors="coerce")
    ok = sdates.notna() & df[arpu_col].notna()
    if ok.sum() == 0: return
    min_month = sdates[ok].min().to_period("M").to_timestamp(how="end")
    max_month = sdates[ok].max().to_period("M").to_timestamp(how="end")
    # extend horizon by 12 months or observed churn horizon
    horizon = 12
    if churn_month_col in df.columns and df[churn_month_col].notna().any():
        horizon = max(horizon, int(df[churn_month_col].max()))
    max_month = max_month + pd.offsets.MonthEnd(horizon)

    idx = pd.date_range(min_month, max_month, freq="ME")
    mrr = pd.Series(0.0, index=idx)

    # accumulate ARPU for each active month per user
    for _, row in df[ok].iterrows():
        start = pd.Timestamp(row[signup_col]).to_period("M").to_timestamp(how="end")
        months_active = int(row[churn_month_col]) if churn_month_col in df.columns and pd.notna(row[churn_month_col]) and row[churn_month_col] > 0 else horizon
        end = start + pd.offsets.MonthEnd(months_active)
        rng = pd.date_range(start, min(end, idx[-1]), freq="ME")
        mrr[rng] = mrr[rng] + float(row[arpu_col])

    fig, ax = plt.subplots()
    ax.plot(mrr.index, mrr.values, linewidth=2.2, marker="o", markersize=3.2)
    ax.set_title("MRR Timeline"); ax.set_xlabel("Month"); ax.set_ylabel("MRR ($)")
    _nice_time_axis(ax, mrr.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 7) Churn rate by country (bar) — robust
def plot_country_churn_rate(
    df: pd.DataFrame,
    country_col: str = "country",
    churned_col: str = "churned",
    out_dir: str = "charts",
    outfile: str = "7_country_churn_rate.png",
    top_n: int = 10,
    min_signups: int = 1
):
    if country_col not in df.columns or churned_col not in df.columns:
        return False

    countries = df[country_col].astype(str).str.strip().str.title()
    churned = pd.to_numeric(df[churned_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    g = pd.DataFrame({country_col: countries, churned_col: churned})
    g = g[g[country_col].notna()]
    if g.empty:
        return False

    agg = g.groupby(country_col, dropna=True).agg(
        churn_rate=(churned_col, "mean"),
        signups=(churned_col, "size")
    )
    agg = agg[agg["signups"] >= min_signups]
    if agg.empty:
        return False

    # show top_n by churn_rate
    agg = agg.sort_values(["churn_rate", "signups"], ascending=[False, False]).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    agg["churn_rate"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    ax.set_title("Churn Rate by Country (Top)")
    ax.set_xlabel("Country"); ax.set_ylabel("Churn Rate")
    for i, (idx, row) in enumerate(agg.iterrows()):
        ax.text(i, row["churn_rate"] + 0.01, f"n={row['signups']}", ha="center", va="bottom", fontsize=9)
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)
    return True


# 8) ARPU distribution (hist)
def plot_arpu_hist(
    df: pd.DataFrame,
    arpu_col: str = "arpu",
    out_dir: str = CHARTS_DIR,
    outfile: str = "8_arpu_hist.png",
    bins: int = 40
):
    if not _has(df, [arpu_col]): return
    x = pd.to_numeric(df[arpu_col], errors="coerce").dropna()
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, edgecolor="#ffffff", linewidth=0.6, alpha=0.95)
    ax.set_title("ARPU Distribution"); ax.set_xlabel("ARPU ($)"); ax.set_ylabel("Users")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)

# 9) Tenure (months until churn) histogram — churned users only
def plot_tenure_hist(
    df: pd.DataFrame,
    churn_month_col: str = "churn_month",
    churned_col: str = "churned",
    out_dir: str = "charts",
    outfile: str = "9_tenure_hist.png",
    max_bins: int = 12
):
    if churn_month_col not in df.columns or churned_col not in df.columns:
        return False

    churned = pd.to_numeric(df[churned_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    tenure = pd.to_numeric(df[churn_month_col], errors="coerce")

    # keep churned users with positive tenure
    x = tenure[(churned == 1) & (tenure.notna()) & (tenure > 0)].astype(int)
    if x.empty:
        return False

    # Nice integer bins from 1..max observed (cap to max_bins if huge)
    m = int(x.max())
    if m <= max_bins:
        bins = range(1, m + 2)  # inclusive last bin
    else:
        bins = max_bins

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, align="left", rwidth=0.9, edgecolor="#ffffff", linewidth=0.6, color="#4e79a7")
    ax.set_title("Tenure Until Churn (Months) — Churned Users")
    ax.set_xlabel("Months"); ax.set_ylabel("Users")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)
    return True


# 10) LTV by plan (naive) = ARPU / median(monthly_churn_prob)
def plot_ltv_by_plan(
    df: pd.DataFrame,
    plan_col: str = "plan",
    arpu_col: str = "arpu",
    churn_prob_col: str = "monthly_churn_prob",
    out_dir: str = CHARTS_DIR,
    outfile: str = "10_ltv_by_plan.png"
):
    if not _has(df, [plan_col, arpu_col, churn_prob_col]): return
    agg = df.groupby(plan_col).agg(
        arpu=("arpu", "median"),
        p=("monthly_churn_prob", "median")
    ).replace({0: np.nan})
    agg = agg.dropna()
    if agg.empty: return
    agg["ltv"] = agg["arpu"] / agg["p"]
    fig, ax = plt.subplots()
    agg["ltv"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    ax.set_title("Naive LTV by Plan (ARPU / churn_prob)"); ax.set_xlabel("Plan"); ax.set_ylabel("LTV ($)")
    _apply_theme(ax)
    _savefig(out_dir, outfile, fig)
# --- cell ---
# ===================== Run All (with column guards, SaaS churn) =====================

def _has(df, cols):
    return all(c in df.columns for c in cols)

# 1) Signups per period (weekly)
if _has(df, ["signup_date"]):
    plot_signups_by_period(
        df, freq="W", outfile="1_signups_by_week.png", rotate_xticks=45, max_ticks=10
    )

# 2) Plan mix
if _has(df, ["plan"]):
    plot_plan_mix(
        df, outfile="2_plan_mix.png"
    )

# 3) Churn rate by plan (requires: plan, churned)
if _has(df, ["plan", "churned"]):
    plot_churn_rate_by_plan(
        df, outfile="3_churn_rate_by_plan.png", min_signups=1
    )

# 4) ARPU by plan (requires: plan, arpu)
if _has(df, ["plan", "arpu"]):
    plot_arpu_by_plan(
        df, outfile="4_arpu_by_plan.png"
    )

# 5) Retention curves — prefer empirical (plan + churn_month), else theoretical (plan + monthly_churn_prob)
if _has(df, ["plan", "churn_month"]) or _has(df, ["plan", "monthly_churn_prob"]):
    plot_retention_curves(
        df, months=12, outfile="5_retention_curves.png"
    )

# 6) MRR timeline (requires: user_id, signup_date, arpu)
if _has(df, ["user_id", "signup_date", "arpu"]):
    plot_mrr_timeline(
        df, outfile="6_mrr_timeline.png", freq="ME", rotate_xticks=45, max_ticks=12
    )

# 7) Country churn rate (requires: country, churned)
if _has(df, ["country", "churned"]):
    plot_country_churn_rate(
        df, outfile="7_country_churn_rate.png", top_n=10, min_signups=1
    )

# 8) ARPU histogram (requires: arpu)
if _has(df, ["arpu"]):
    plot_arpu_hist(
        df, outfile="8_arpu_hist.png", bins=40
    )

# 9) Tenure histogram (requires: churn_month, churned)
if _has(df, ["churn_month", "churned"]):
    plot_tenure_hist(
        df, outfile="9_tenure_hist.png", max_bins=12
    )

# 10) Naive LTV by plan (requires: plan, arpu, monthly_churn_prob)
if _has(df, ["plan", "arpu", "monthly_churn_prob"]):
    plot_ltv_by_plan(
        df, outfile="10_ltv_by_plan.png"
    )
