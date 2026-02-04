from __future__ import annotations

from io import BytesIO
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _read_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    filename = (filename or "").lower()
    bio = BytesIO(file_bytes)

    if filename.endswith(".csv"):
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


def parse_date_with_audit(
    df: pd.DataFrame,
    col: str,
    *,
    dayfirst_fallback: bool = True,
    fallback_mode: str = "mixed",
) -> tuple[pd.Series, str]:
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

    parsed = None
    audit = ""
    if fallback_mode == "mixed":
        try:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            audit = f"{col}: fallback via pd.to_datetime(format='mixed')"
        except TypeError:
            parsed = None
    if parsed is None:
        parsed = pd.to_datetime(s, errors="coerce")
        audit = f"{col}: fallback via pd.to_datetime (dateutil parser)"

    if dayfirst_fallback and parsed.isna().mean() > 0 and s_str.str.contains(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", regex=True).any():
        parsed2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed, audit = parsed2, f"{col}: fallback via pd.to_datetime (dayfirst=True)"

    return parsed, audit


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].str.lower().isin(["nan", "none", "null", ""]), col] = pd.NA

    for col in out.columns:
        if not _is_date_like_col(col):
            continue
        if out[col].dtype != "object":
            continue
        parsed, _ = parse_date_with_audit(out, col)
        if parsed.notna().mean() >= 0.70:
            out[col] = parsed

    for col in out.columns:
        if out[col].dtype == "object":
            coerced = pd.to_numeric(out[col].str.replace(",", "", regex=False), errors="coerce")
            if coerced.notna().mean() >= 0.70:
                out[col] = coerced

    return out


def _basic_clean(df: pd.DataFrame, *, fill_missing: bool = True) -> pd.DataFrame:
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


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _coerce_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
