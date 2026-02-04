from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_processing_cases import (
    _apply_ecommerce_rules,
    _apply_marketing_rules,
    _apply_saas_rules,
    _make_charts,
)
from .data_processing_io import (
    _basic_clean,
    _coerce_types,
    _infer_dataset_type,
    _normalize_columns,
    _read_dataframe,
)


# This module is a Django-friendly port of the "SimpleDataProcessing" notebook pattern:
# load -> normalize -> clean -> summarize -> produce charts -> export cleaned CSV.
#
# It supports three common demo "cases":
# - Ecommerce sales
# - SaaS churn
# - Marketing performance
#
# If you want 1:1 parity with your notebooks, paste/port the exact notebook
# transformations into the per-case sections in data_processing_cases.py.


@dataclass
class ReportResult:
    dataset_type_selected: str
    dataset_type_inferred: str
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    head_html: str
    stats_html: str | None
    notes: list[str]
    charts: list[dict]
    cleaned_csv_bytes: bytes


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
