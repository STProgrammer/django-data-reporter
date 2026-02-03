from django import forms

DATASET_CHOICES = [
    ("auto", "Auto-detect (recommended)"),
    ("ecommerce_sales", "Case 1 — Ecommerce sales"),
    ("saas_churn", "Case 2 — SaaS churn"),
    ("marketing_perf", "Case 3 — Marketing performance"),
]

class UploadDataForm(forms.Form):
    dataset_type = forms.ChoiceField(
        choices=DATASET_CHOICES,
        required=True,
        initial="auto",
        label="Dataset type",
        help_text="Pick Auto-detect unless you know the dataset case."
    )

    data_file = forms.FileField(
        label="Upload CSV or Excel (.csv, .xlsx)",
        help_text="Max 20 MB"
    )

    def clean_data_file(self):
        f = self.cleaned_data["data_file"]
        name = (f.name or "").lower()
        if not (name.endswith(".csv") or name.endswith(".xlsx")):
            raise forms.ValidationError("Please upload a .csv or .xlsx file.")
        return f
