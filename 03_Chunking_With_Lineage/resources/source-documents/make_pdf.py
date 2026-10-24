"""Write password-reset-comparison.pdf, the one-page sheet Lab 03 Part A reads. Not a lab dependency.

Run from this folder:  uv run --no-project --with fpdf2 python make_pdf.py
Every fact on the sheet restates the five Atlas/Boreal policies in data/support-case.json (constructed for the labs).
The creation date is fixed, so the same fpdf2 version writes the same bytes.
"""
from datetime import datetime, timezone

from fpdf import FPDF

pdf = FPDF(format="A4")
pdf.set_creation_date(datetime(2026, 10, 24, tzinfo=timezone.utc))
pdf.set_title("Password reset: Atlas and Boreal")
pdf.set_margins(20, 20, 20)
pdf.add_page()

pdf.set_font("Helvetica", "B", 18)
pdf.cell(text="Password reset: Atlas and Boreal", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.cell(text="Comparison sheet for the support team", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

pdf.set_font("Helvetica", "", 11)
with pdf.table(col_widths=(40, 60, 60), text_align="LEFT") as table:
    for row in [("Product", "Reset method", "Link valid for"),
                ("Atlas", "Email", "30 minutes"),
                ("Boreal", "Mobile app", "10 minutes")]:
        cells = table.row()
        for text in row:
            cells.cell(text)
pdf.ln(8)

pdf.set_font("Helvetica", "B", 13)
pdf.cell(text="Boreal", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6, "Boreal reset links expire after 10 minutes. Password reset uses a mobile app.")
pdf.ln(4)
pdf.set_font("Helvetica", "I", 10)
pdf.multi_cell(0, 6, "For Atlas, see the Atlas help page.")

pdf.output("password-reset-comparison.pdf")
print("wrote password-reset-comparison.pdf")
