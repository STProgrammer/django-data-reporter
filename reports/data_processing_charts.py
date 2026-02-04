from __future__ import annotations

from io import BytesIO
import base64
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _chart_from_fig(title: str, fig) -> dict:
    return {"title": title, "png_base64": _fig_to_base64(fig)}


def _has(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


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


def _chart_top_categories(
    df: pd.DataFrame,
    cat_col: str,
    value_col: Optional[str] = None,
    title: str = "",
) -> Optional[dict]:
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
    if n >= 120:
        freq = "ME"
    elif n >= 40:
        freq = "W"
    else:
        freq = "D"

    ts = s.set_index(date_col).resample(freq)[value_col].sum()
    fig = plt.figure(figsize=(7.5, 3.6))
    ax = fig.add_subplot(111)
    ts.plot(ax=ax)
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
