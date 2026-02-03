from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# This module is a Django-friendly port of the "SimpleDataProcessing" notebook pattern:
# load → normalize → clean → summarize → produce charts → export cleaned CSV.
#
# It supports three common demo "cases":
# - Ecommerce sales
# - SaaS churn
# - Marketing performance
#
# If you want 1:1 parity with your notebooks, paste/port the exact notebook
# transformations into the per-case sections below (functions starting with _case_*).
# The web UI + reporting glue will stay the same.


@dataclass
class ReportResult:
    dataset_type_selected: str          # what the user picked ("auto" or a case)
    dataset_type_inferred: str          # what the app inferred after reading columns
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    head_html: str
    stats_html: str | None
    notes: list[str]
    charts: list[dict]                 # [{title, png_base64}]
    cleaned_csv_bytes: bytes


# ------------------------
# IO + cleaning utilities
# ------------------------

def _read_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    filename = (filename or "").lower()
    bio = BytesIO(file_bytes)

    if filename.endswith(".csv"):
        # Try UTF-8 first, fall back to latin1.
        try:
            return pd.read_csv(bio)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, encoding="latin1")

    if filename.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")

    raise ValueError("Unsupported file type. Please upload .csv or .xlsx")


def _norm_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_norm_col(c) for c in out.columns]
    return out


_CANDIDATE_FORMATS = [
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
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y %H:%M:%S",
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%Y.%m.%d",
    "%Y.%m.%d %H:%M:%S",
]

_EPOCH_10_RE = re.compile(r"^\s*\d{10}\s*$")
_EPOCH_13_RE = re.compile(r"^\s*\d{13}\s*$")
_EPOCH_16_RE = re.compile(r"^\s*\d{16}\s*$")
_DATE_NAME_RE = re.compile(r"(?:^|_)(date|datetime|timestamp|time)(?:$|_)", re.I)


def _is_date_like_col(name: str) -> bool:
    n = name.lower()
    if _DATE_NAME_RE.search(n):
        return True
    if n.endswith("_at"):
        return True
    return False


def _guess_epoch_unit(sample: pd.Series, min_match: float = 0.7) -> Optional[str]:
    s = sample.dropna()
    if s.empty:
        return None
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


def _heuristic_filter_formats(strings: pd.Series) -> list[str]:
    sample = strings.dropna().astype(str)
    if sample.empty:
        return _CANDIDATE_FORMATS

    text = " ".join(sample.head(50))
    has_t = "T" in text
    has_slash = "/" in text
    has_dash = "-" in text
    has_dot = "." in text
    has_tz = "+" in text or "Z" in text

    candidates: list[str] = []
    for fmt in _CANDIDATE_FORMATS:
        if has_t and "T" not in fmt:
            continue
        if not has_t and "T" in fmt:
            continue
        if has_slash and "/" not in fmt:
            continue
        if has_dash and "-" not in fmt and "T" not in fmt:
            continue
        if has_dot and "." not in fmt and "%f" not in fmt:
            pass
        if has_tz and "%z" not in fmt and "T" in fmt:
            pass
        candidates.append(fmt)

    if len(candidates) < 3:
        return _CANDIDATE_FORMATS
    return candidates


def _guess_datetime_format(series: pd.Series, sample_size: int = 200, min_match: float = 0.7) -> Optional[str]:
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
            if best_score == 1.0:
                break
    return best_fmt if best_score >= min_match else None


def parse_date_with_audit(df: pd.DataFrame, col: str, *, dayfirst_fallback: bool = True) -> tuple[pd.Series, str]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found")

    s = df[col]
    if s.dtype.kind in "Mm":
        return s, f"{col}: already datetime64[ns]"

    if s.isna().all():
        return pd.to_datetime(s, errors="coerce"), f"{col}: all values null -> all NaT"

    unit = _guess_epoch_unit(s if s.dtype == "O" else s.astype("object"))
    if unit:
        return pd.to_datetime(s, unit=unit, errors="coerce"), f"{col}: parsed as UNIX epoch ({unit})"

    fmt = _guess_datetime_format(s)
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce"), f"{col}: parsed with explicit format '{fmt}'"

    s_str = s.astype(str)
    if s_str.str.contains("Z|\\+\\d{2}:?\\d{2}", regex=True).any():
        for iso_fmt in ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            parsed = pd.to_datetime(s, format=iso_fmt, errors="coerce")
            if parsed.notna().any():
                return parsed, f"{col}: parsed with explicit ISO+TZ format '{iso_fmt}'"

    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    audit = f"{col}: fallback via pd.to_datetime(format='mixed')"

    if dayfirst_fallback and parsed.isna().mean() > 0 and s_str.str.contains(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", regex=True).any():
        parsed2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed, audit = parsed2, f"{col}: fallback via pd.to_datetime (dayfirst=True)"

    return parsed, audit


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Best-effort dtype inference:
    - Try to parse datetimes for columns that look like dates.
    - Try to parse numerics from object columns.
    '''
    out = df.copy()

    # First: strip strings and normalize missing tokens
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].str.lower().isin(["nan", "none", "null", ""]), col] = pd.NA

    # Datetime candidates by name
    for col in out.columns:
        if not _is_date_like_col(col):
            continue
        if out[col].dtype != "object":
            continue
        parsed, _ = parse_date_with_audit(out, col)
        if parsed.notna().mean() >= 0.70:
            out[col] = parsed

    # Numeric coercion for remaining object columns
    for col in out.columns:
        if out[col].dtype == "object":
            coerced = pd.to_numeric(out[col].str.replace(",", "", regex=False), errors="coerce")
            # accept if meaningful share converts
            if coerced.notna().mean() >= 0.70:
                out[col] = coerced

    return out


def _basic_clean(df: pd.DataFrame, *, fill_missing: bool = True) -> pd.DataFrame:
    '''
    Conservative cleaning steps (similar to SimpleDataProcessing notebooks):
    - drop empty rows/cols
    - drop duplicates
    - fill numeric NA with median; categorical NA with mode/"unknown"
    '''
    out = df.copy()

    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    out = out.drop_duplicates()

    if fill_missing:
        num_cols = list(out.select_dtypes(include=["number"]).columns)
        for col in num_cols:
            if out[col].isna().any():
                median = out[col].median()
                out[col] = out[col].fillna(median)

        other_cols = [c for c in out.columns if c not in num_cols]
        for col in other_cols:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) else "unknown"
                out[col] = out[col].fillna(fill_value)

    return out


_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab",
]


def _colors(n: int) -> list[str]:
    base = list(_PALETTE)
    return (base * ((n + len(base) - 1) // len(base)))[:n]


def _apply_theme(ax) -> None:
    ax.set_facecolor("#f9fafb")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _normalize_freq(freq: str) -> str:
    if not isinstance(freq, str):
        return freq
    freq = freq.upper()
    mapping = {
        "M": "ME",
        "MS": "MS",
        "Q": "QE",
        "QS": "QS",
        "Y": "YE",
        "A": "YE",
        "AS": "YS",
    }
    return mapping.get(freq, freq)


def _nice_time_axis(ax, idx, freq: str, max_ticks: int = 10, rotate: int = 0) -> None:
    freq = _normalize_freq(freq)
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(idx)
        except Exception:
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

    fmt = None
    if freq == "D":
        fmt = mdates.DateFormatter("%Y-%m-%d")
    elif freq == "W":
        fmt = mdates.DateFormatter("%G-W%V")
    elif freq in ("MS", "ME"):
        fmt = mdates.DateFormatter("%Y-%m")
    elif freq in ("QS", "QE"):
        labels = [f"{d.year}-Q{((d.month - 1) // 3) + 1}" for d in tick_pos]
        ax.set_xticklabels(labels)
    else:
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


def _handle_missing(df: pd.DataFrame, col: str, action: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    a = action.strip().lower()

    def _is_num(s) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def _const_value(raw: str, series: pd.Series):
        raw = raw.strip()
        if raw.lower() in {"nan", "na", "null"}:
            return np.nan
        if _is_num(series):
            try:
                return float(raw)
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(raw, errors="coerce")
            except Exception:
                pass
        return raw

    if a in {"set_zero", "zero"}:
        df[col] = df[col].fillna(0)
        return df
    if a in {"set_nan", "nan"}:
        df[col] = df[col].where(~df[col].isna(), np.nan)
        return df
    if a in {"drop_row", "drop"}:
        return df[~df[col].isna()].copy()
    if a == "mean":
        if _is_num(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        return df
    if a == "median":
        if _is_num(df[col]):
            df[col] = df[col].fillna(df[col].median())
        return df
    if a == "mode":
        if df[col].isna().any():
            m = df[col].mode(dropna=True)
            if not m.empty:
                df[col] = df[col].fillna(m.iloc[0])
        return df
    if a in {"ffill", "bfill"}:
        df[col] = df[col].fillna(method=a)
        return df

    if a.startswith("const:"):
        raw = action.split(":", 1)[1]
        val = _const_value(raw, df[col])
        df[col] = df[col].fillna(val)
        return df

    if a.startswith("percentile:"):
        if _is_num(df[col]):
            try:
                q = float(action.split(":", 1)[1])
                q = min(max(q, 0.0), 1.0)
                val = df[col].quantile(q)
                df[col] = df[col].fillna(val)
            except Exception:
                pass
        return df

    if a.startswith("mean_by:") or a.startswith("median_by:") or a.startswith("mode_by:"):
        try:
            method, grp = a.split(":", 1)
            if grp in df.columns and df[col].isna().any():
                if method == "mean_by" and _is_num(df[col]):
                    fill_vals = df.groupby(grp)[col].transform("mean")
                    df[col] = df[col].fillna(fill_vals)
                elif method == "median_by" and _is_num(df[col]):
                    fill_vals = df.groupby(grp)[col].transform("median")
                    df[col] = df[col].fillna(fill_vals)
                elif method == "mode_by":
                    def _grp_mode(s):
                        m = s.mode(dropna=True)
                        return m.iloc[0] if not m.empty else np.nan
                    fill_vals = df.groupby(grp)[col].transform(_grp_mode)
                    df[col] = df[col].fillna(fill_vals)
        except Exception:
            pass
        return df

    if a.startswith("interpolate:"):
        method = action.split(":", 1)[1].strip() or "linear"
        if _is_num(df[col]):
            try:
                if method == "time" and not isinstance(df.index, pd.DatetimeIndex):
                    method = "linear"
                df[col] = df[col].interpolate(method=method, limit_direction="both")
            except Exception:
                pass
        return df

    return df


def _normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        to_replace=["", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null", "-"],
        value=np.nan,
    )


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _chart_from_fig(title: str, fig) -> dict:
    return {"title": title, "png_base64": _fig_to_base64(fig)}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = list(df.columns)
    cand = [_norm_col(c) for c in candidates]

    for c in cand:
        if c in cols:
            return c

    for c in cols:
        for token in cand:
            if token and token in c:
                return c

    return None


def _has(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def _infer_dataset_type(df: pd.DataFrame, filename: str = "") -> str:
    cols = set(df.columns)
    fn = (filename or "").lower()

    marketing_tokens = {"impressions", "clicks", "spend", "campaign", "conversions", "ctr", "cpc", "roas"}
    if (len(cols.intersection(marketing_tokens)) >= 2) or any(t in fn for t in ["marketing", "campaign", "ads"]):
        return "marketing_perf"

    churn_tokens = {"churn", "churned", "is_churned", "mrr", "arr", "plan", "subscription", "tenure"}
    if (len(cols.intersection(churn_tokens)) >= 2) or any(t in fn for t in ["churn", "saas", "subscription"]):
        return "saas_churn"

    ecommerce_tokens = {"order", "order_date", "sales", "revenue", "quantity", "qty", "product", "category"}
    if (len(cols.intersection(ecommerce_tokens)) >= 2) or any(t in fn for t in ["ecommerce", "sales", "orders"]):
        return "ecommerce_sales"

    return "generic"


# ------------------------
# Chart helpers
# ------------------------

def _chart_missing(df: pd.DataFrame) -> list[dict]:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) == 0:
        return []
    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_subplot(111)
    missing.head(12).plot(kind="bar", ax=ax)
    ax.set_title("Missing values per column (top 12)")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    return [{"title": "Missing values", "png_base64": _fig_to_base64(fig)}]


def _chart_numeric_distributions(df: pd.DataFrame, max_cols: int = 3) -> list[dict]:
    charts: list[dict] = []
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    for col in num_cols[:max_cols]:
        fig = plt.figure(figsize=(7.5, 3.6))
        ax = fig.add_subplot(111)
        s = df[col].dropna()
        if len(s) > 0:
            ax.hist(s, bins=30)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        charts.append({"title": f"Distribution: {col}", "png_base64": _fig_to_base64(fig)})
    return charts


def _chart_top_categories(df: pd.DataFrame, cat_col: str, value_col: Optional[str] = None, title: str = "") -> Optional[dict]:
    if cat_col not in df.columns:
        return None

    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        top = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).head(10)
        ylabel = f"Sum({value_col})"
    else:
        top = df[cat_col].astype(str).value_counts().head(10)
        ylabel = "Count"

    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_subplot(111)
    top.plot(kind="bar", ax=ax)
    ax.set_title(title or f"Top {cat_col}")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    return {"title": title or f"Top {cat_col}", "png_base64": _fig_to_base64(fig)}


def _chart_time_series_sum(df: pd.DataFrame, date_col: str, value_col: str, title: str) -> Optional[dict]:
    if date_col not in df.columns or value_col not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return None

    s = df[[date_col, value_col]].dropna().sort_values(date_col)
    if s.empty:
        return None

    n = len(s)
    if n >= 365:
        g = s.groupby(s[date_col].dt.to_period("M"))[value_col].sum()
        x = g.index.to_timestamp()
    else:
        g = s.groupby(s[date_col].dt.to_period("D"))[value_col].sum()
        x = g.index.to_timestamp()

    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(x, g.values)
    ax.set_title(title)
    ax.set_xlabel(date_col)
    ax.set_ylabel(f"Sum({value_col})")
    return {"title": title, "png_base64": _fig_to_base64(fig)}


def _chart_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Optional[dict]:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        return None
    s = df[[x_col, y_col]].dropna()
    if s.empty:
        return None
    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(s[x_col], s[y_col], s=12)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return {"title": title, "png_base64": _fig_to_base64(fig)}


def _chart_qty_by_period(
    df: pd.DataFrame,
    date_col: str,
    qty_col: str,
    freq: str = "W",
    rotate_xticks: int = 45,
    max_ticks: int = 10,
) -> Optional[dict]:
    if not _has(df, [date_col, qty_col]):
        return None
    freq = _normalize_freq(freq)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    qty = pd.to_numeric(df[qty_col], errors="coerce")
    ok = dates.notna() & qty.notna()
    if ok.sum() == 0:
        return None
    ts = (
        pd.DataFrame({date_col: dates[ok], qty_col: qty[ok]})
        .set_index(date_col)
        .resample(freq)[qty_col]
        .sum()
    )
    if ts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 6 if freq == "W" else 20 if freq in ("MS", "ME") else 0.9
    ax.bar(ts.index, ts.values, width=width, color=_colors(len(ts)), edgecolor="#ffffff", linewidth=0.6)
    label = "Day" if freq == "D" else "Week" if freq == "W" else "Month"
    title = f"Quantity per {label}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_revenue_by_period(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    freq: str = "ME",
    kind: str = "line",
    rotate_xticks: int = 0,
    max_ticks: int = 10,
) -> Optional[dict]:
    if not _has(df, [date_col, revenue_col]):
        return None
    freq = _normalize_freq(freq)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    rev = pd.to_numeric(df[revenue_col], errors="coerce")
    ok = dates.notna() & rev.notna()
    if ok.sum() == 0:
        return None
    ts = (
        pd.DataFrame({date_col: dates[ok], revenue_col: rev[ok]})
        .set_index(date_col)
        .resample(freq)[revenue_col]
        .sum()
    )
    if ts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    if kind == "bar":
        width = 6 if freq == "W" else 20 if freq in ("MS", "ME") else 0.9
        ax.bar(ts.index, ts.values, width=width, color=_colors(len(ts)), edgecolor="#ffffff", linewidth=0.6)
    else:
        ax.plot(ts.index, ts.values, linewidth=2.2, marker="o", markersize=4)
    label = "Day" if freq == "D" else "Week" if freq == "W" else "Month"
    title = f"Revenue per {label}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_top_countries_by_revenue(
    df: pd.DataFrame,
    country_col: str,
    revenue_col: str,
    top_n: int = 5,
) -> Optional[dict]:
    if not _has(df, [country_col, revenue_col]):
        return None
    s = df.groupby(country_col)[revenue_col].sum().sort_values(ascending=False).head(top_n)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    ax.set_title("Top Countries by Revenue")
    ax.set_xlabel("Country")
    ax.set_ylabel("Revenue")
    for p in ax.patches:
        p.set_linewidth(0.6)
    _apply_theme(ax)
    return _chart_from_fig("Top Countries by Revenue", fig)


def _chart_unitprice_vs_revenue(
    df: pd.DataFrame,
    price_col: str,
    revenue_col: str,
) -> Optional[dict]:
    if not _has(df, [price_col, revenue_col]):
        return None
    x = pd.to_numeric(df[price_col], errors="coerce")
    y = pd.to_numeric(df[revenue_col], errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x[ok], y[ok], alpha=0.7, s=28, linewidths=0.5, edgecolors="#ffffff")
    ax.set_yscale("log")
    title = "Unit Price vs Revenue (log scale)"
    ax.set_title(title)
    ax.set_xlabel("Unit Price")
    ax.set_ylabel("Revenue (log)")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_revenue_by_product_boxplot(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    top_n: int = 10,
) -> Optional[dict]:
    if not _has(df, [product_col, revenue_col]):
        return None
    totals = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(top_n)
    if totals.empty:
        return None
    top_idx = totals.index
    sub = df[df[product_col].isin(top_idx)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sub.boxplot(column=revenue_col, by=product_col, rot=45, ax=ax, patch_artist=True)
    colors = _colors(len(top_idx))
    for i, box in enumerate(ax.artists):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor("#ffffff")
        box.set_linewidth(0.6)
    title = "Revenue per Order by Product (Top 10)"
    ax.set_title(title)
    plt.suptitle("")
    ax.set_xlabel("Product")
    ax.set_ylabel("Revenue per Order")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_category_share_pie(
    df: pd.DataFrame,
    category_col: str,
    revenue_col: str,
) -> Optional[dict]:
    if not _has(df, [category_col, revenue_col]):
        return None
    s = df.groupby(category_col)[revenue_col].sum()
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        s.values,
        labels=s.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=_colors(len(s)),
        wedgeprops={"edgecolor": "#ffffff", "linewidth": 0.8},
        textprops={"fontsize": 10},
    )
    title = "Revenue Share by Category"
    ax.set_title(title)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_price_vs_qty_scatter(
    df: pd.DataFrame,
    price_col: str,
    qty_col: str,
) -> Optional[dict]:
    if not _has(df, [price_col, qty_col]):
        return None
    x = pd.to_numeric(df[price_col], errors="coerce")
    y = pd.to_numeric(df[qty_col], errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x[ok], y[ok], alpha=0.7, marker="x", s=28)
    title = "Unit Price vs Quantity"
    ax.set_title(title)
    ax.set_xlabel("Unit Price")
    ax.set_ylabel("Quantity")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_daily_orders(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    rotate_xticks: int = 0,
    max_ticks: int = 12,
) -> Optional[dict]:
    if not _has(df, [date_col, id_col]):
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    ids = df[id_col]
    ok = dates.notna() & ids.notna()
    if ok.sum() == 0:
        return None
    ts = (
        pd.DataFrame({date_col: dates[ok], id_col: ids[ok]})
        .groupby(pd.Grouper(key=date_col, freq="D"))[id_col]
        .nunique()
    )
    if ts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts.index, ts.values, linewidth=2.0, marker="o", markersize=4)
    title = "Daily Order Count"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders")
    _nice_time_axis(ax, ts.index, freq="D", max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_corr_heatmap_wo_ids(
    df: pd.DataFrame,
    id_suffixes: tuple[str, ...] = ("id",),
    uniq_ratio_cut: float = 0.9,
) -> Optional[dict]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = []
    for c in num_cols:
        cl = c.lower()
        if any(cl.endswith(suf) for suf in id_suffixes):
            drop_cols.append(c)
            continue
        unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > uniq_ratio_cut:
            drop_cols.append(c)
    use_cols = [c for c in num_cols if c not in drop_cols]
    if len(use_cols) < 2:
        return None
    corr = df[use_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(corr.values, interpolation="nearest", cmap="viridis")
    ax.set_title("Correlation Heatmap (IDs excluded)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    cbar = fig.colorbar(img)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=12)
    _apply_theme(ax)
    return _chart_from_fig("Correlation Heatmap (IDs excluded)", fig)


def _chart_revenue_hist(
    df: pd.DataFrame,
    revenue_col: str,
    bins: int = 40,
) -> Optional[dict]:
    if revenue_col not in df.columns:
        return None
    rev = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
    if rev.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rev, bins=bins, edgecolor="#ffffff", linewidth=0.6, alpha=0.9)
    title = "Revenue Distribution per Order"
    ax.set_title(title)
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Count")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_signups_by_period(
    df: pd.DataFrame,
    date_col: str,
    freq: str = "W",
    rotate_xticks: int = 45,
    max_ticks: int = 10,
) -> Optional[dict]:
    if not _has(df, [date_col]):
        return None
    freq = _normalize_freq(freq)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    ok = dates.notna()
    if ok.sum() == 0:
        return None
    ts = pd.DataFrame({date_col: dates[ok]}).set_index(date_col).resample(freq).size()
    if ts.empty:
        return None
    fig, ax = plt.subplots()
    width = 6 if freq == "W" else 20 if freq in ("MS", "ME") else 0.9
    ax.bar(ts.index, ts.values, width=width, color=_colors(len(ts)), edgecolor="#ffffff", linewidth=0.6)
    label = "Day" if freq == "D" else "Week" if freq == "W" else "Month"
    title = f"New Signups per {label}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Signups")
    _nice_time_axis(ax, ts.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_plan_mix(df: pd.DataFrame, plan_col: str) -> Optional[dict]:
    if not _has(df, [plan_col]):
        return None
    s = df[plan_col].value_counts()
    if s.empty:
        return None
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    title = "Plan Mix"
    ax.set_title(title)
    ax.set_xlabel("Plan")
    ax.set_ylabel("Users")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_churn_rate_by_plan(
    df: pd.DataFrame,
    plan_col: str,
    churned_col: str,
    min_signups: int = 1,
) -> Optional[dict]:
    if not _has(df, [plan_col, churned_col]):
        return None
    plans = df[plan_col].astype(str).str.strip()
    churned = pd.to_numeric(df[churned_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    mask = plans.notna()
    g = pd.DataFrame({plan_col: plans[mask], churned_col: churned[mask]})
    if g.empty:
        return None
    agg = g.groupby(plan_col, dropna=True).agg(
        churn_rate=(churned_col, "mean"),
        signups=(churned_col, "size"),
    )
    agg = agg[agg["signups"] >= min_signups].sort_values("churn_rate", ascending=False)
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    agg["churn_rate"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    title = "Churn Rate by Plan"
    ax.set_title(title)
    ax.set_xlabel("Plan")
    ax.set_ylabel("Churn Rate")
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(i, row["churn_rate"] + 0.01, f"n={row['signups']}", ha="center", va="bottom", fontsize=9)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_arpu_by_plan(df: pd.DataFrame, plan_col: str, arpu_col: str) -> Optional[dict]:
    if not _has(df, [plan_col, arpu_col]):
        return None
    s = df.groupby(plan_col)[arpu_col].mean()
    if s.empty:
        return None
    fig, ax = plt.subplots()
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    title = "ARPU by Plan"
    ax.set_title(title)
    ax.set_xlabel("Plan")
    ax.set_ylabel("ARPU ($)")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_retention_curves(
    df: pd.DataFrame,
    plan_col: str,
    churn_prob_col: str,
    churn_month_col: str,
    months: int = 12,
) -> Optional[dict]:
    if not _has(df, [plan_col]):
        return None
    fig, ax = plt.subplots()
    plotted = False
    if churn_month_col in df.columns:
        plans = df[plan_col].unique()
        colors = _colors(len(plans))
        for i, (plan, g) in enumerate(df.groupby(plan_col)):
            cm = g[churn_month_col].fillna(0).clip(lower=0).astype(int).to_numpy()
            surv = []
            for m in range(0, months + 1):
                survived = ((cm == 0) | (cm > m)).mean()
                surv.append(survived)
            ax.plot(
                range(0, months + 1),
                surv,
                label=str(plan),
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                color=colors[i] if len(colors) > 1 else colors[0],
            )
            plotted = True
    elif churn_prob_col in df.columns:
        plans = df[plan_col].unique()
        colors = _colors(len(plans))
        for i, (plan, g) in enumerate(df.groupby(plan_col)):
            p = g[churn_prob_col].dropna().median()
            if pd.isna(p):
                continue
            surv = [(1 - p) ** m for m in range(0, months + 1)]
            ax.plot(
                range(0, months + 1),
                surv,
                label=str(plan),
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                color=colors[i] if len(colors) > 1 else colors[0],
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    title = "Retention Curves by Plan (0-12 months)"
    ax.set_title(title)
    ax.set_xlabel("Months since Signup")
    ax.set_ylabel("Retention")
    ax.legend(frameon=False)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_mrr_timeline(
    df: pd.DataFrame,
    user_col: str,
    signup_col: str,
    churn_month_col: str,
    arpu_col: str,
    freq: str = "ME",
    rotate_xticks: int = 45,
    max_ticks: int = 12,
) -> Optional[dict]:
    if not _has(df, [user_col, signup_col, arpu_col]):
        return None
    freq = _normalize_freq(freq)
    sdates = pd.to_datetime(df[signup_col], errors="coerce")
    ok = sdates.notna() & df[arpu_col].notna()
    if ok.sum() == 0:
        return None
    min_month = sdates[ok].min().to_period("M").to_timestamp(how="end")
    max_month = sdates[ok].max().to_period("M").to_timestamp(how="end")
    horizon = 12
    if churn_month_col in df.columns and df[churn_month_col].notna().any():
        horizon = max(horizon, int(df[churn_month_col].max()))
    max_month = max_month + pd.offsets.MonthEnd(horizon)
    idx = pd.date_range(min_month, max_month, freq="ME")
    mrr = pd.Series(0.0, index=idx)
    for _, row in df[ok].iterrows():
        start = pd.Timestamp(row[signup_col]).to_period("M").to_timestamp(how="end")
        months_active = (
            int(row[churn_month_col])
            if churn_month_col in df.columns and pd.notna(row[churn_month_col]) and row[churn_month_col] > 0
            else horizon
        )
        end = start + pd.offsets.MonthEnd(months_active)
        rng = pd.date_range(start, min(end, idx[-1]), freq="ME")
        mrr[rng] = mrr[rng] + float(row[arpu_col])
    if mrr.empty:
        return None
    fig, ax = plt.subplots()
    ax.plot(mrr.index, mrr.values, linewidth=2.2, marker="o", markersize=3.2)
    title = "MRR Timeline"
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("MRR ($)")
    _nice_time_axis(ax, mrr.index, freq=freq, max_ticks=max_ticks, rotate=rotate_xticks)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_country_churn_rate(
    df: pd.DataFrame,
    country_col: str,
    churned_col: str,
    top_n: int = 10,
    min_signups: int = 1,
) -> Optional[dict]:
    if not _has(df, [country_col, churned_col]):
        return None
    countries = df[country_col].astype(str).str.strip().str.title()
    churned = pd.to_numeric(df[churned_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    g = pd.DataFrame({country_col: countries, churned_col: churned})
    g = g[g[country_col].notna()]
    if g.empty:
        return None
    agg = g.groupby(country_col, dropna=True).agg(
        churn_rate=(churned_col, "mean"),
        signups=(churned_col, "size"),
    )
    agg = agg[agg["signups"] >= min_signups]
    if agg.empty:
        return None
    agg = agg.sort_values(["churn_rate", "signups"], ascending=[False, False]).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    agg["churn_rate"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    title = "Churn Rate by Country (Top)"
    ax.set_title(title)
    ax.set_xlabel("Country")
    ax.set_ylabel("Churn Rate")
    for i, (_, row) in enumerate(agg.iterrows()):
        ax.text(i, row["churn_rate"] + 0.01, f"n={row['signups']}", ha="center", va="bottom", fontsize=9)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_arpu_hist(df: pd.DataFrame, arpu_col: str, bins: int = 40) -> Optional[dict]:
    if arpu_col not in df.columns:
        return None
    x = pd.to_numeric(df[arpu_col], errors="coerce").dropna()
    if x.empty:
        return None
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, edgecolor="#ffffff", linewidth=0.6, alpha=0.95)
    title = "ARPU Distribution"
    ax.set_title(title)
    ax.set_xlabel("ARPU ($)")
    ax.set_ylabel("Users")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_tenure_hist(
    df: pd.DataFrame,
    churn_month_col: str,
    churned_col: str,
    max_bins: int = 12,
) -> Optional[dict]:
    if not _has(df, [churn_month_col, churned_col]):
        return None
    churned = pd.to_numeric(df[churned_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    tenure = pd.to_numeric(df[churn_month_col], errors="coerce")
    x = tenure[(churned == 1) & (tenure.notna()) & (tenure > 0)].astype(int)
    if x.empty:
        return None
    m = int(x.max())
    bins = range(1, m + 2) if m <= max_bins else max_bins
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, align="left", rwidth=0.9, edgecolor="#ffffff", linewidth=0.6, color="#4e79a7")
    title = "Tenure Until Churn (Months) - Churned Users"
    ax.set_title(title)
    ax.set_xlabel("Months")
    ax.set_ylabel("Users")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_ltv_by_plan(
    df: pd.DataFrame,
    plan_col: str,
    arpu_col: str,
    churn_prob_col: str,
) -> Optional[dict]:
    if not _has(df, [plan_col, arpu_col, churn_prob_col]):
        return None
    agg = (
        df.groupby(plan_col)
        .agg(arpu=(arpu_col, "median"), p=(churn_prob_col, "median"))
        .replace({0: np.nan})
        .dropna()
    )
    if agg.empty:
        return None
    agg["ltv"] = agg["arpu"] / agg["p"]
    fig, ax = plt.subplots()
    agg["ltv"].plot(kind="bar", ax=ax, color=_colors(len(agg)), edgecolor="#ffffff")
    title = "Naive LTV by Plan (ARPU / churn_prob)"
    ax.set_title(title)
    ax.set_xlabel("Plan")
    ax.set_ylabel("LTV ($)")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_total_spend_by_channel(df: pd.DataFrame, channel_col: str, spend_col: str) -> Optional[dict]:
    if not _has(df, [channel_col, spend_col]):
        return None
    s = df.groupby(channel_col, dropna=False)[spend_col].sum().sort_values(ascending=False)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.2))
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    title = "Total Spend by Channel"
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Spend")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_avg_roas_by_channel(df: pd.DataFrame, channel_col: str, roas_col: str) -> Optional[dict]:
    if not _has(df, [channel_col, roas_col]):
        return None
    s = df.dropna(subset=[roas_col]).groupby(channel_col, dropna=False)[roas_col].mean().sort_values(ascending=False)
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.2))
    s.plot(kind="bar", ax=ax, color=_colors(len(s)), edgecolor="#ffffff")
    title = "Average ROAS by Channel"
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("ROAS")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_cvr_boxplot(df: pd.DataFrame, channel_col: str, cvr_col: str) -> Optional[dict]:
    if not _has(df, [channel_col, cvr_col]):
        return None
    d = df.dropna(subset=[cvr_col])
    if d.empty:
        return None
    order = d.groupby(channel_col)[cvr_col].median().sort_values(ascending=False).index
    data = [d.loc[d[channel_col] == ch, cvr_col].to_numpy(dtype=float) for ch in order]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=order)
    cols = _colors(len(order))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=cols[i], edgecolor="#ffffff", linewidth=0.8, alpha=0.95)
    for item in bp["whiskers"] + bp["caps"]:
        item.set(color="#666666", linewidth=0.8)
    for med in bp["medians"]:
        med.set(color="#1f1f1f", linewidth=1.4)
    title = "CVR by Channel"
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("CVR")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_spend_timeseries_by_channel(
    df: pd.DataFrame,
    date_col: str,
    channel_col: str,
    spend_col: str,
    last_n_days: Optional[int] = 60,
) -> Optional[dict]:
    if not _has(df, [date_col, channel_col, spend_col]):
        return None
    d = df.copy()
    if last_n_days is not None and not d[date_col].isna().all():
        cutoff = d[date_col].max() - pd.Timedelta(days=last_n_days - 1)
        d = d[d[date_col] >= cutoff]
    ts = (
        d.groupby([date_col, channel_col], as_index=False)[spend_col]
        .sum()
        .pivot(index=date_col, columns=channel_col, values=spend_col)
        .sort_index()
    )
    if ts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cols = _colors(len(ts.columns))
    for i, ch in enumerate(ts.columns):
        ax.plot(ts.index, ts[ch], label=ch, linewidth=2.0, marker="o", markersize=3.2, color=cols[i])
    title = f"Spend by Channel{' (Last ' + str(last_n_days) + ' Days)' if last_n_days else ''}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spend")
    ax.legend(ncol=3, frameon=False)
    _nice_time_axis(ax, ts.index.to_numpy(), freq="D", max_ticks=9, rotate=0)
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


def _chart_revenue_vs_spend_scatter(
    df: pd.DataFrame,
    channel_col: str,
    spend_col: str,
    revenue_col: str,
) -> Optional[dict]:
    if not _has(df, [channel_col, spend_col, revenue_col]):
        return None
    agg = (
        df.groupby(channel_col, dropna=False)
        .agg(spend=(spend_col, "sum"), revenue=(revenue_col, "sum"))
        .reset_index()
    )
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agg["spend"], agg["revenue"], marker="o", s=80, alpha=0.9, edgecolors="#ffffff", linewidths=0.6, color="#4e79a7")
    for _, r in agg.iterrows():
        ax.annotate(
            str(r[channel_col]),
            (r["spend"], r["revenue"]),
            xytext=(6, 4),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.75),
        )
    title = "Revenue vs Spend by Channel"
    ax.set_title(title)
    ax.set_xlabel("Spend")
    ax.set_ylabel("Revenue")
    _apply_theme(ax)
    return _chart_from_fig(title, fig)


# ------------------------
# Per-case pipelines
# ------------------------

def _case_ecommerce(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    charts: list[dict] = []

    date_col = _find_column(df, ["order_date", "orderdate", "date", "purchase_date", "transaction_date", "invoice_date"])
    qty_col = _find_column(df, ["quantity", "qty", "units", "unit_sold", "items"])
    revenue_col = _find_column(df, ["revenue", "sales", "sales_amount", "amount", "total", "total_revenue", "total_sales", "net_sales", "gross_sales", "order_total"])
    country_col = _find_column(df, ["country", "region", "nation"])
    price_col = _find_column(df, ["unit_price", "price", "unitprice", "unit_cost"])
    product_col = _find_column(df, ["product", "product_name", "productname", "item", "sku"])
    category_col = _find_column(df, ["category", "product_category", "productcategory", "segment", "department", "product_type"])
    order_id_col = _find_column(df, ["order_id", "orderid", "invoice_id", "transaction_id", "id"])

    if date_col and qty_col:
        c = _chart_qty_by_period(df, date_col, qty_col, freq="W", rotate_xticks=45, max_ticks=8)
        if c:
            charts.append(c)
            notes.append(f"Quantity by period uses date={date_col}, qty={qty_col}")

    if date_col and revenue_col:
        c = _chart_revenue_by_period(df, date_col, revenue_col, freq="ME", kind="line", rotate_xticks=0, max_ticks=10)
        if c:
            charts.append(c)
            notes.append(f"Revenue by period uses date={date_col}, revenue={revenue_col}")

    if country_col and revenue_col:
        c = _chart_top_countries_by_revenue(df, country_col, revenue_col, top_n=5)
        if c:
            charts.append(c)

    if price_col and revenue_col:
        c = _chart_unitprice_vs_revenue(df, price_col, revenue_col)
        if c:
            charts.append(c)

    if product_col and revenue_col:
        c = _chart_revenue_by_product_boxplot(df, product_col, revenue_col, top_n=10)
        if c:
            charts.append(c)

    if category_col and revenue_col:
        c = _chart_category_share_pie(df, category_col, revenue_col)
        if c:
            charts.append(c)

    if price_col and qty_col:
        c = _chart_price_vs_qty_scatter(df, price_col, qty_col)
        if c:
            charts.append(c)

    if date_col and order_id_col:
        c = _chart_daily_orders(df, date_col, order_id_col, rotate_xticks=0, max_ticks=12)
        if c:
            charts.append(c)

    c = _chart_corr_heatmap_wo_ids(df, id_suffixes=("id",), uniq_ratio_cut=0.9)
    if c:
        charts.append(c)

    if revenue_col:
        c = _chart_revenue_hist(df, revenue_col, bins=40)
        if c:
            charts.append(c)

    return charts, notes


def _case_saas_churn(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    charts: list[dict] = []

    date_col = _find_column(df, ["signup_date", "signupdate", "created_at", "created", "date", "start_date", "startdate"])
    plan_col = _find_column(df, ["plan", "tier", "subscription", "package", "pricing_plan", "plan_name"])
    churn_col = _find_column(df, ["churned", "is_churned", "churn", "canceled", "cancelled", "is_cancelled", "is_canceled", "cancelled_flag"])
    arpu_col = _find_column(df, ["arpu", "avg_revenue_per_user", "average_revenue_per_user"])
    churn_prob_col = _find_column(df, ["monthly_churn_prob", "churn_prob", "monthly_churn_probability"])
    churn_month_col = _find_column(df, ["churn_month", "tenure", "months_until_churn"])
    user_col = _find_column(df, ["user_id", "userid", "customer_id"])
    country_col = _find_column(df, ["country", "region"])

    if date_col:
        c = _chart_signups_by_period(df, date_col, freq="W", rotate_xticks=45, max_ticks=10)
        if c:
            charts.append(c)

    if plan_col:
        c = _chart_plan_mix(df, plan_col)
        if c:
            charts.append(c)

    if plan_col and churn_col:
        c = _chart_churn_rate_by_plan(df, plan_col, churn_col, min_signups=1)
        if c:
            charts.append(c)

    if plan_col and arpu_col:
        c = _chart_arpu_by_plan(df, plan_col, arpu_col)
        if c:
            charts.append(c)

    if plan_col and (churn_month_col or churn_prob_col):
        c = _chart_retention_curves(
            df,
            plan_col=plan_col,
            churn_prob_col=churn_prob_col or "monthly_churn_prob",
            churn_month_col=churn_month_col or "churn_month",
            months=12,
        )
        if c:
            charts.append(c)

    if user_col and date_col and arpu_col:
        c = _chart_mrr_timeline(
            df,
            user_col=user_col,
            signup_col=date_col,
            churn_month_col=churn_month_col or "churn_month",
            arpu_col=arpu_col,
            freq="ME",
            rotate_xticks=45,
            max_ticks=12,
        )
        if c:
            charts.append(c)

    if country_col and churn_col:
        c = _chart_country_churn_rate(df, country_col, churn_col, top_n=10, min_signups=1)
        if c:
            charts.append(c)

    if arpu_col:
        c = _chart_arpu_hist(df, arpu_col, bins=40)
        if c:
            charts.append(c)

    if churn_month_col and churn_col:
        c = _chart_tenure_hist(df, churn_month_col, churn_col, max_bins=12)
        if c:
            charts.append(c)

    if plan_col and arpu_col and churn_prob_col:
        c = _chart_ltv_by_plan(df, plan_col, arpu_col, churn_prob_col)
        if c:
            charts.append(c)

    return charts, notes


def _case_marketing(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    charts: list[dict] = []

    date_col = _find_column(df, ["date", "day"])
    channel_col = _find_column(df, ["channel", "campaign", "campaign_name", "source", "platform", "ad_set"])
    spend_col = _find_column(df, ["spend", "cost", "ad_spend", "ad_cost", "amount_spent"])
    clicks_col = _find_column(df, ["clicks", "click", "link_clicks"])
    conv_col = _find_column(df, ["conversions", "conversion", "purchases", "leads", "orders", "signups"])
    revenue_col = _find_column(df, ["revenue", "sales", "value", "purchase_value", "conversion_value"])
    roas_col = _find_column(df, ["roas"])
    cvr_col = _find_column(df, ["cvr"])

    if channel_col and spend_col:
        c = _chart_total_spend_by_channel(df, channel_col, spend_col)
        if c:
            charts.append(c)

    if channel_col and roas_col:
        c = _chart_avg_roas_by_channel(df, channel_col, roas_col)
        if c:
            charts.append(c)

    if channel_col and cvr_col:
        c = _chart_cvr_boxplot(df, channel_col, cvr_col)
        if c:
            charts.append(c)

    if date_col and channel_col and spend_col:
        c = _chart_spend_timeseries_by_channel(df, date_col, channel_col, spend_col, last_n_days=60)
        if c:
            charts.append(c)

    if channel_col and spend_col and revenue_col:
        c = _chart_revenue_vs_spend_scatter(df, channel_col, spend_col, revenue_col)
        if c:
            charts.append(c)

    return charts, notes


def _apply_ecommerce_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _normalize_missing_tokens(out)
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()

    qty_col = _find_column(out, ["qty", "quantity", "units", "unit_sold", "items"])
    price_col = _find_column(out, ["unit_price", "price", "unitprice", "unit_cost"])
    revenue_col = _find_column(out, ["revenue", "sales", "sales_amount", "amount", "total", "total_revenue", "total_sales", "net_sales", "gross_sales", "order_total"])
    product_col = _find_column(out, ["product", "product_name", "productname", "item", "sku"])
    category_col = _find_column(out, ["category", "product_category", "productcategory", "segment", "department", "product_type"])
    country_col = _find_column(out, ["country", "region", "nation"])

    if qty_col and price_col and revenue_col:
        mask = out[qty_col].isna() & out[revenue_col].notna() & out[price_col].notna() & (out[price_col] != 0)
        out.loc[mask, qty_col] = out.loc[mask, revenue_col] / out.loc[mask, price_col]
        mask = out[price_col].isna() & out[revenue_col].notna() & out[qty_col].notna() & (out[qty_col] != 0)
        out.loc[mask, price_col] = out.loc[mask, revenue_col] / out.loc[mask, qty_col]
        mask = out[revenue_col].isna() & out[qty_col].notna() & out[price_col].notna()
        out.loc[mask, revenue_col] = out.loc[mask, qty_col] * out.loc[mask, price_col]

    if price_col and product_col and out[price_col].isna().any():
        out = _handle_missing(out, price_col, f"median_by:{product_col}")

    for keep_nan_col in [country_col, product_col, category_col]:
        if keep_nan_col:
            out = _handle_missing(out, keep_nan_col, "set_nan")

    numeric_exclude = {c for c in [qty_col, price_col, revenue_col] if c}
    for c in out.select_dtypes(include=[np.number]).columns:
        if c not in numeric_exclude and out[c].isna().any():
            out = _handle_missing(out, c, "median")

    object_exclude = {c for c in [country_col, product_col, category_col] if c}
    for c in out.select_dtypes(include=["object"]).columns:
        if c not in object_exclude and out[c].isna().any():
            out = _handle_missing(out, c, "mode")

    return out


def _apply_saas_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _normalize_missing_tokens(out)

    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()

    if "signup_date" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["signup_date"]):
        parsed, _ = parse_date_with_audit(out, "signup_date")
        out["signup_date"] = parsed

    for c in ["plan", "country"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.title()
            out = _handle_missing(out, c, "set_nan")

    for c in ["monthly_churn_prob", "churn_month", "arpu"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "churn_month" in out.columns:
        cm = pd.to_numeric(out["churn_month"], errors="coerce")
        out["churned"] = (cm.fillna(0) > 0).astype(int)
        out["churn_month"] = cm.fillna(0).clip(lower=0).round().astype(int)
    elif "churned" in out.columns:
        out["churned"] = pd.to_numeric(out["churned"], errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    if "monthly_churn_prob" in out.columns:
        out["monthly_churn_prob"] = out["monthly_churn_prob"].clip(lower=0)
    if "arpu" in out.columns:
        out["arpu"] = out["arpu"].clip(lower=0)

    out = out.drop_duplicates()
    numeric_exclude = {"churn_month"}
    for c in out.select_dtypes(include=[np.number]).columns:
        if c not in numeric_exclude and out[c].isna().any():
            out = _handle_missing(out, c, "median")

    return out


def _apply_marketing_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _normalize_missing_tokens(out)

    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()

    if "channel" in out.columns:
        out["channel"] = out["channel"].astype("string").str.title()
        out = _handle_missing(out, "channel", "set_nan")

    if "date" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["date"]):
        parsed, _ = parse_date_with_audit(out, "date")
        out["date"] = parsed

    for c in ["spend", "clicks", "conversions", "revenue"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "date" in out.columns and "channel" in out.columns:
        out = out[out["date"].notna() & out["channel"].notna()].copy()

    if _has(out, ["spend", "clicks"]):
        out["cpc"] = np.where(out["clicks"] > 0, out["spend"] / out["clicks"], np.nan)
    if _has(out, ["clicks", "conversions"]):
        out["cvr"] = np.where(out["clicks"] > 0, out["conversions"] / out["clicks"], np.nan)
    if _has(out, ["spend", "revenue"]):
        out["roas"] = np.where(out["spend"] > 0, out["revenue"] / out["spend"], np.nan)

    out = out.drop_duplicates()
    return out


def _make_charts(df: pd.DataFrame, *, chosen: str, inferred: str) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    charts: list[dict] = []

    if chosen == "ecommerce_sales":
        extra, n = _case_ecommerce(df)
        charts.extend(extra)
        notes.extend(n)
    elif chosen == "saas_churn":
        extra, n = _case_saas_churn(df)
        charts.extend(extra)
        notes.extend(n)
    elif chosen == "marketing_perf":
        extra, n = _case_marketing(df)
        charts.extend(extra)
        notes.extend(n)
    else:
        charts.extend(_chart_missing(df))

    if not charts:
        charts.extend(_chart_missing(df))
        charts.extend(_chart_numeric_distributions(df, max_cols=3))

    if len(charts) < 5 and chosen == "generic":
        charts.extend(_chart_numeric_distributions(df, max_cols=3))
        if len(charts) < 5:
            obj_cols = list(df.select_dtypes(include=["object"]).columns)
            if obj_cols:
                c = _chart_top_categories(df, obj_cols[0], None, f"Top values: {obj_cols[0]}")
                if c:
                    charts.append(c)

    return charts, notes


def build_report(file_bytes: bytes, filename: str, dataset_type: str = "auto") -> ReportResult:
    df = _read_dataframe(file_bytes, filename)
    original_shape = df.shape

    df = _normalize_columns(df)
    df = _coerce_types(df)
    inferred = _infer_dataset_type(df, filename=filename)
    chosen = inferred if dataset_type == "auto" else dataset_type
    cleaned = _basic_clean(df, fill_missing=(chosen == "generic"))

    if chosen == "ecommerce_sales":
        cleaned = _apply_ecommerce_rules(cleaned)
    elif chosen == "saas_churn":
        cleaned = _apply_saas_rules(cleaned)
    elif chosen == "marketing_perf":
        cleaned = _apply_marketing_rules(cleaned)
    cleaned_shape = cleaned.shape

    dtypes = {c: str(cleaned[c].dtype) for c in cleaned.columns}
    missing = {c: int(cleaned[c].isna().sum()) for c in cleaned.columns}

    head_html = cleaned.head(20).to_html(index=False, classes="table")
    stats_html = None
    num = cleaned.select_dtypes(include=["number"])
    if num.shape[1] > 0:
        stats_html = num.describe().T.reset_index().to_html(index=False, classes="table")

    charts, notes = _make_charts(cleaned, chosen=chosen, inferred=inferred)
    cleaned_csv_bytes = cleaned.to_csv(index=False).encode("utf-8")

    return ReportResult(
        dataset_type_selected=dataset_type,
        dataset_type_inferred=inferred,
        original_shape=original_shape,
        cleaned_shape=cleaned_shape,
        columns=list(cleaned.columns),
        dtypes=dtypes,
        missing=missing,
        head_html=head_html,
        stats_html=stats_html,
        notes=notes,
        charts=charts,
        cleaned_csv_bytes=cleaned_csv_bytes,
    )
