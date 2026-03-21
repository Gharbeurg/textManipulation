# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import re
import fitz  # PyMuPDF

PDF_DIR = Path(r"C:/PYTHON/.entree/Sources")
TXT_DIR_OUT = Path(r"C:/PYTHON/.data/ResultatsPDF")

# ===============================
# LOG
# ===============================

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")


# ===============================
# FILTRES INTELLIGENTS
# ===============================

def is_metadata_line(text):
    t = text.lower()

    patterns = [
        "doi",
        "http",
        "www.",
        "copyright",
        "all rights reserved",
        "licensed under",
        "peer reviewed version",
        "correspondence",
        "email:",
    ]

    return any(p in t for p in patterns)


def is_reference_line(text):
    return bool(re.match(r"^\[\d+\]|^[A-Z][a-z]+, [A-Z]\.", text))


def is_equation(text):
    return bool(re.search(r"[=<>±∑∫]", text))


def is_table_complex(text):
    # table très technique (factor loadings, coefficients, etc.)
    keywords = ["loading", "coefficient", "regression", "std.", "p-value"]
    return any(k in text.lower() for k in keywords)


def is_short_noise(text):
    return len(text) < 40 and text.isupper()


# ===============================
# EXTRACTION SIMPLE
# ===============================

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        blocks = page.get_text("blocks")

        # tri lecture naturelle
        blocks.sort(key=lambda b: (b[1], b[0]))

        for b in blocks:
            text = b[4].strip()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


# ===============================
# NETTOYAGE PRINCIPAL
# ===============================

def clean_text(text):

    lines = text.split("\n")
    cleaned = []

    seen_abstract = False
    in_references = False

    for line in lines:
        l = line.strip()

        if not l:
            continue

        # STOP REFERENCES
        if re.match(r"references|bibliography", l, re.I):
            break

        # METADATA
        if is_metadata_line(l):
            continue

        # REFERENCES lignes individuelles
        if is_reference_line(l):
            continue

        # EQUATIONS
        if is_equation(l):
            continue

        # NOISE
        if is_short_noise(l):
            continue

        # ABSTRACT doublon
        if re.match(r"abstract", l, re.I):
            if seen_abstract:
                continue
            seen_abstract = True
            cleaned.append("ABSTRACT")
            continue

        # SUMMARY => ABSTRACT
        if re.match(r"summary", l, re.I):
            cleaned.append("ABSTRACT")
            continue

        # TABLE
        if re.match(r"table", l, re.I):
            cleaned.append(f"TABLE: {l}")
            continue

        # FIGURE supprimée
        if re.match(r"(figure|fig\.)", l, re.I):
            continue

        # TABLE complexe
        if is_table_complex(l):
            continue

        cleaned.append(l)

    return "\n".join(cleaned)


# ===============================
# FINAL CLEAN
# ===============================

def normalize(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===============================
# PIPELINE
# ===============================

def process_pdf(pdf_path):

    log(f"Processing {pdf_path.name}")

    raw = extract_text_blocks(pdf_path)
    cleaned = clean_text(raw)
    final = normalize(cleaned)

    out_path = TXT_DIR_OUT / f"{pdf_path.stem}.txt"
    TXT_DIR_OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final, encoding="utf-8")

    log(f"Saved → {out_path}")


# ===============================
# MAIN
# ===============================

def main():

    log("START")

    pdfs = list(PDF_DIR.glob("*.pdf"))

    for pdf in pdfs:
        process_pdf(pdf)

    log("DONE")


if __name__ == "__main__":
    main()