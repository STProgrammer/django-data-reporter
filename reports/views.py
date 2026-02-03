from __future__ import annotations
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from .forms import UploadDataForm
from .data_processing import build_report, ReportResult

@require_http_methods(["GET", "POST"])
def upload_view(request):
    report: ReportResult | None = None

    if request.method == "POST":
        form = UploadDataForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_type = form.cleaned_data["dataset_type"]
            f = form.cleaned_data["data_file"]
            file_bytes = f.read()

            report = build_report(
                file_bytes=file_bytes,
                filename=f.name,
                dataset_type=dataset_type,
            )

            request.session["cleaned_csv"] = report.cleaned_csv_bytes.decode("utf-8")
            request.session["uploaded_name"] = f.name
    else:
        form = UploadDataForm()

    return render(request, "reports/upload.html", {"form": form, "report": report})

def download_cleaned_csv(request):
    csv_text = request.session.get("cleaned_csv")
    uploaded_name = request.session.get("uploaded_name", "cleaned.csv")
    if not csv_text:
        return redirect("upload")

    safe_base = uploaded_name.rsplit(".", 1)[0] if "." in uploaded_name else uploaded_name
    filename = f"{safe_base}_cleaned.csv"

    resp = HttpResponse(csv_text, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp
