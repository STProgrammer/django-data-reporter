from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data_processing_io import (
    _coerce_numeric_cols,
    _handle_missing,
    _normalize_missing_tokens,
    _strip_object_cols,
    parse_date_with_audit,
)
from .data_processing_charts import (
    _chart_arpu_by_plan,
    _chart_arpu_hist,
    _chart_avg_roas_by_channel,
    _chart_category_share_pie,
    _chart_corr_heatmap_wo_ids,
    _chart_cvr_boxplot,
    _chart_daily_orders,
    _chart_ltv_by_plan,
    _chart_missing,
    _chart_mrr_timeline,
    _chart_numeric_distributions,
    _chart_plan_mix,
    _chart_price_vs_qty_scatter,
    _chart_qty_by_period,
    _chart_revenue_by_period,
    _chart_revenue_hist,
    _chart_revenue_by_product_boxplot,
    _chart_revenue_vs_spend_scatter,
    _chart_signups_by_period,
    _chart_spend_timeseries_by_channel,
    _chart_top_categories,
    _chart_top_countries_by_revenue,
    _chart_tenure_hist,
    _chart_total_spend_by_channel,
    _chart_unitprice_vs_revenue,
    _chart_country_churn_rate,
    _chart_churn_rate_by_plan,
    _chart_retention_curves,
    _has,
)


def _append_chart(
    charts: list[dict],
    chart: Optional[dict],
    *,
    notes: Optional[list[str]] = None,
    note: Optional[str] = None,
) -> None:
    if chart:
        charts.append(chart)
        if notes is not None and note:
            notes.append(note)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = list(df.columns)
    cand = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()) for c in candidates]

    for c in cand:
        if c in cols:
            return c

    for c in cols:
        for token in cand:
            if token and token in c:
                return c

    return None


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
        _append_chart(
            charts,
            _chart_qty_by_period(df, date_col, qty_col, freq="W", rotate_xticks=45, max_ticks=8),
            notes=notes,
            note=f"Quantity by period uses date={date_col}, qty={qty_col}",
        )

    if date_col and revenue_col:
        _append_chart(
            charts,
            _chart_revenue_by_period(df, date_col, revenue_col, freq="ME", kind="line", rotate_xticks=0, max_ticks=10),
            notes=notes,
            note=f"Revenue by period uses date={date_col}, revenue={revenue_col}",
        )

    if country_col and revenue_col:
        _append_chart(charts, _chart_top_countries_by_revenue(df, country_col, revenue_col, top_n=5))

    if price_col and revenue_col:
        _append_chart(charts, _chart_unitprice_vs_revenue(df, price_col, revenue_col))

    if product_col and revenue_col:
        _append_chart(charts, _chart_revenue_by_product_boxplot(df, product_col, revenue_col, top_n=10))

    if category_col and revenue_col:
        _append_chart(charts, _chart_category_share_pie(df, category_col, revenue_col))

    if price_col and qty_col:
        _append_chart(charts, _chart_price_vs_qty_scatter(df, price_col, qty_col))

    if date_col and order_id_col:
        _append_chart(charts, _chart_daily_orders(df, date_col, order_id_col, rotate_xticks=0, max_ticks=12))

    _append_chart(charts, _chart_corr_heatmap_wo_ids(df, id_suffixes=("id",), uniq_ratio_cut=0.9))

    if revenue_col:
        _append_chart(charts, _chart_revenue_hist(df, revenue_col, bins=40))

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
        _append_chart(charts, _chart_signups_by_period(df, date_col, freq="W", rotate_xticks=45, max_ticks=10))

    if plan_col:
        _append_chart(charts, _chart_plan_mix(df, plan_col))

    if plan_col and churn_col:
        _append_chart(charts, _chart_churn_rate_by_plan(df, plan_col, churn_col, min_signups=1))

    if plan_col and arpu_col:
        _append_chart(charts, _chart_arpu_by_plan(df, plan_col, arpu_col))

    if plan_col and (churn_month_col or churn_prob_col):
        _append_chart(
            charts,
            _chart_retention_curves(
                df,
                plan_col=plan_col,
                churn_prob_col=churn_prob_col or "monthly_churn_prob",
                churn_month_col=churn_month_col or "churn_month",
                months=12,
            ),
        )

    if user_col and date_col and arpu_col:
        _append_chart(
            charts,
            _chart_mrr_timeline(
                df,
                user_col=user_col,
                signup_col=date_col,
                churn_month_col=churn_month_col or "churn_month",
                arpu_col=arpu_col,
                freq="ME",
                rotate_xticks=45,
                max_ticks=12,
            ),
        )

    if country_col and churn_col:
        _append_chart(charts, _chart_country_churn_rate(df, country_col, churn_col, top_n=10, min_signups=1))

    if arpu_col:
        _append_chart(charts, _chart_arpu_hist(df, arpu_col, bins=40))

    if churn_month_col and churn_col:
        _append_chart(charts, _chart_tenure_hist(df, churn_month_col, churn_col, max_bins=12))

    if plan_col and arpu_col and churn_prob_col:
        _append_chart(charts, _chart_ltv_by_plan(df, plan_col, arpu_col, churn_prob_col))

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
        _append_chart(charts, _chart_total_spend_by_channel(df, channel_col, spend_col))

    if channel_col and roas_col:
        _append_chart(charts, _chart_avg_roas_by_channel(df, channel_col, roas_col))

    if channel_col and cvr_col:
        _append_chart(charts, _chart_cvr_boxplot(df, channel_col, cvr_col))

    if date_col and channel_col and spend_col:
        _append_chart(charts, _chart_spend_timeseries_by_channel(df, date_col, channel_col, spend_col, last_n_days=60))

    if channel_col and spend_col and revenue_col:
        _append_chart(charts, _chart_revenue_vs_spend_scatter(df, channel_col, spend_col, revenue_col))

    return charts, notes


def _apply_ecommerce_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _normalize_missing_tokens(out)
    out = _strip_object_cols(out)

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
    out = _strip_object_cols(out)

    if "signup_date" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["signup_date"]):
        parsed, _ = parse_date_with_audit(out, "signup_date", fallback_mode="dateutil")
        out["signup_date"] = parsed

    for c in ["plan", "country"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.title()
            out = _handle_missing(out, c, "set_nan")

    out = _coerce_numeric_cols(out, ["monthly_churn_prob", "churn_month", "arpu"])

    if "churn_month" in out.columns:
        cm = pd.to_numeric(out["churn_month"], errors="coerce")
        out["churned"] = (cm.fillna(0) > 0).astype(int)
        out["churn_month"] = cm.fillna(0).clip(lower=0).round().astype(int)
    else:
        if "churned" in out.columns:
            del out["churned"]

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

    required = ["date", "channel", "spend", "clicks", "conversions", "revenue"]
    if not _has(out, required):
        missing = sorted(list(set(required) - set(out.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    out = _strip_object_cols(out)

    if "channel" in out.columns:
        out["channel"] = out["channel"].astype("string").str.title()
        out = _handle_missing(out, "channel", "set_nan")

    if "date" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["date"]):
        parsed, _ = parse_date_with_audit(out, "date", fallback_mode="dateutil")
        out["date"] = parsed

    out = _coerce_numeric_cols(out, ["spend", "clicks", "conversions", "revenue"])

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
