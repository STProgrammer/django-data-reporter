# Data Reporter (Django) — Portfolio Demo (SimpleDataProcessing → Web UI)

A small Django app that demonstrates **data automation** in a “client-looking” way:

- Upload a **CSV** (or Excel `.xlsx`)
- Auto-detect the dataset case (or pick it manually)
- Run a consistent cleaning pipeline (normalize → type inference → dedupe → fill missing)
- Show a preview table, numeric summary, and **charts/diagrams on the page**
- Download the cleaned dataset as CSV

This app is meant for **local demo on Windows** and as a portfolio item.

## Supported demo cases

The UI supports three “case-style” datasets (matching the style of your `SimpleDataProcessing` repo):

1) **Ecommerce sales**
2) **SaaS churn**
3) **Marketing performance**

You can upload your **three CSV files** from the repo and the UI will generate case-relevant charts when it finds the expected columns.

> If you want 1:1 parity with your notebook logic: keep the UI as-is and paste your exact per-case notebook transforms into `reports/data_processing_cases.py` (functions `_case_ecommerce`, `_case_saas_churn`, `_case_marketing`).

## Quick start (Windows)

1) Open PowerShell in the project folder

2) Build a virtual environment named `.venv` and activate it:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Run migrations (uses SQLite locally):

```powershell
py manage.py migrate
```

5) Start the server:

```powershell
py manage.py runserver
```

6) Open:

- http://127.0.0.1:8000/

Upload your CSV (from `SimpleDataProcessing/case1_*`, `case2_*`, `case3_*`).

## Notes / design

- Charts are rendered server-side using Matplotlib (headless).
- Column names are normalized automatically (spaces → underscores, lowercase).
- File upload limit is capped at 20 MB in settings (good for demos).
