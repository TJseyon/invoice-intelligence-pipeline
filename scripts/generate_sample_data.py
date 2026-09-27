"""
Generates a set of synthetic invoice images plus a ground-truth CSV.

Why synthetic instead of scraped real invoices: no network access is needed
to build or demo this project, there are zero licensing/PII concerns, and
because we render the text ourselves we have exact, indisputable ground
truth for every field -- a legitimate stand-in for "manually verified"
values at this stage. The README is explicit that swapping in real,
manually-verified scanned invoices is the natural next step before citing
these numbers as a production claim.

Produces three groups, matching the edge cases the pipeline explicitly
handles:
  - normal/          : clean, readable invoices (majority of the set)
  - blank/           : intentionally blank or near-blank pages -> empty_document
  - non_english/     : invoices rendered in French -> unsupported_language
    (with the default OCR_LANGUAGES=eng config)
  - noisy/           : rotated + low-contrast, to exercise preprocessing and
    produce some genuinely low-confidence / malformed extractions

Usage (run from the repo root):
    python scripts/generate_sample_data.py --out backend/eval/data --count 30
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

random.seed(42)

VENDORS = [
    "Acme Supply Co", "Blue Ridge Hardware", "Northwind Traders", "Summit Logistics",
    "Cascade Office Supplies", "Ironclad Manufacturing", "Riverside Consulting",
    "Pinnacle IT Services", "Harbor Freight & Parts", "Meridian Print Shop",
]
ITEMS = [
    ("Widget A", 12.50), ("Widget B", 8.00), ("Consulting Hour", 150.00),
    ("Shipping Pallet", 45.00), ("Printer Cartridge", 22.75), ("Office Chair", 89.99),
    ("Steel Bracket", 3.20), ("Software License", 199.00), ("Safety Gloves (box)", 14.60),
    ("Labor - Install", 75.00),
]

FRENCH_LINES = [
    "FACTURE", "Fournisseur: Établissements Dubois", "Date: 2026-03-14",
    "Numéro de facture: FR-2291", "Article: Câble électrique", "Quantité: 4",
    "Prix unitaire: 9,50", "Total: 210,40 EUR", "Merci de votre confiance",
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_invoice(vendor, inv_num, inv_date, currency, items, tax_rate=0.08):
    subtotal = round(sum(qty * price for _, qty, price in items), 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    img = Image.new("RGB", (900, 1100), "white")
    d = ImageDraw.Draw(img)
    f_title, f_body, f_small = _font(28), _font(20), _font(16)

    y = 40
    d.text((40, y), "INVOICE", font=f_title, fill="black"); y += 50
    d.text((40, y), f"VENDOR: {vendor}", font=f_body, fill="black"); y += 30
    d.text((40, y), f"INVOICE #: {inv_num}", font=f_body, fill="black"); y += 30
    d.text((40, y), f"DATE: {inv_date}", font=f_body, fill="black"); y += 30
    d.text((40, y), f"CURRENCY: {currency}", font=f_body, fill="black"); y += 50

    d.text((40, y), "LINE ITEMS", font=f_body, fill="black"); y += 30
    for desc, qty, price in items:
        line_total = round(qty * price, 2)
        line = f"ITEM: {desc}  QTY: {qty}  PRICE: ${price:.2f}  LINETOTAL: ${line_total:.2f}"
        d.text((60, y), line, font=f_small, fill="black")
        y += 26

    y += 20
    d.text((40, y), f"SUBTOTAL: ${subtotal:.2f}", font=f_body, fill="black"); y += 30
    d.text((40, y), f"TAX: ${tax:.2f}", font=f_body, fill="black"); y += 30
    d.text((40, y), f"TOTAL: ${total:.2f}", font=f_title, fill="black"); y += 50

    return img, subtotal, tax, total


def generate(out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    n_normal = int(count * 0.7)
    n_blank = max(2, int(count * 0.1))
    n_non_english = max(2, int(count * 0.1))
    n_noisy = count - n_normal - n_blank - n_non_english

    idx = 0

    # --- normal, clean, readable invoices ---
    for _ in range(n_normal):
        idx += 1
        vendor = random.choice(VENDORS)
        inv_num = f"INV-{random.randint(1000, 9999)}"
        inv_date = f"2026-{random.randint(1,9):02d}-{random.randint(1,28):02d}"
        n_items = random.randint(1, 4)
        chosen = random.sample(ITEMS, n_items)
        items = [(desc, random.randint(1, 5), price) for desc, price in chosen]

        img, subtotal, tax, total = _draw_invoice(vendor, inv_num, inv_date, "USD", items)
        fname = f"normal_{idx:03d}.png"
        img.save(out_dir / fname)

        rows.append({
            "filename": fname, "category": "normal",
            "vendor_name": vendor, "invoice_number": inv_num, "invoice_date": inv_date,
            "currency": "USD", "subtotal_amount": subtotal, "tax_amount": tax, "total_amount": total,
            "expected_status": "ok_or_flagged", "notes": "",
        })

    # --- blank / unreadable ---
    for i in range(n_blank):
        idx += 1
        img = Image.new("RGB", (900, 1100), "white")
        if i % 2 == 1:  # a couple with just a stray mark, not perfectly blank
            d = ImageDraw.Draw(img)
            d.line((100, 100, 120, 120), fill="lightgray", width=2)
        fname = f"blank_{idx:03d}.png"
        img.save(out_dir / fname)
        rows.append({
            "filename": fname, "category": "blank",
            "vendor_name": "", "invoice_number": "", "invoice_date": "",
            "currency": "", "subtotal_amount": "", "tax_amount": "", "total_amount": "",
            "expected_status": "empty_document", "notes": "intentionally blank page",
        })

    # --- non-English (French) ---
    for _ in range(n_non_english):
        idx += 1
        img = Image.new("RGB", (900, 1100), "white")
        d = ImageDraw.Draw(img)
        f_body = _font(22)
        y = 60
        for line in FRENCH_LINES:
            d.text((40, y), line, font=f_body, fill="black")
            y += 36
        fname = f"french_{idx:03d}.png"
        img.save(out_dir / fname)
        rows.append({
            "filename": fname, "category": "non_english",
            "vendor_name": "Établissements Dubois", "invoice_number": "FR-2291",
            "invoice_date": "2026-03-14", "currency": "EUR",
            "subtotal_amount": "", "tax_amount": "", "total_amount": 210.40,
            "expected_status": "unsupported_language", "notes": "French text; default OCR config is English-only",
        })

    # --- noisy / rotated / low-contrast (malformed extraction candidates) ---
    for _ in range(n_noisy):
        idx += 1
        vendor = random.choice(VENDORS)
        inv_num = f"INV-{random.randint(1000, 9999)}"
        inv_date = f"2026-{random.randint(1,9):02d}-{random.randint(1,28):02d}"
        chosen = random.sample(ITEMS, 2)
        items = [(desc, random.randint(1, 5), price) for desc, price in chosen]
        img, subtotal, tax, total = _draw_invoice(vendor, inv_num, inv_date, "USD", items)

        img = img.rotate(random.choice([-7, -4, 4, 7]), fillcolor="white", expand=False)
        # Low contrast: blend toward gray
        gray = Image.new("RGB", img.size, (180, 180, 180))
        img = Image.blend(img, gray, alpha=0.45)

        fname = f"noisy_{idx:03d}.png"
        img.save(out_dir / fname)
        rows.append({
            "filename": fname, "category": "noisy",
            "vendor_name": vendor, "invoice_number": inv_num, "invoice_date": inv_date,
            "currency": "USD", "subtotal_amount": subtotal, "tax_amount": tax, "total_amount": total,
            "expected_status": "flagged_or_failed", "notes": "rotated + low contrast: OCR confidence should be lower",
        })

    csv_path = out_dir / "eval_set.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} sample documents -> {out_dir}")
    print(f"Ground truth CSV -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="backend/eval/data")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    generate(Path(args.out), args.count)
