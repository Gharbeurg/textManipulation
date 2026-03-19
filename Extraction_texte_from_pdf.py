import re
import fitz  # PyMuPDF
from pathlib import Path
from collections import Counter

# =========================
# PARAMÈTRES
# =========================

PDF_DIR = Path(r"C:/PYTHON/.entree/Sources")
PDF_DIR_OUT = Path(r"C:/PYTHON/.data/ResultatsPDF")

TOP_MARGIN_RATIO = 0.10
BOTTOM_MARGIN_RATIO = 0.10
REPEAT_LINE_MIN_PAGES_RATIO = 0.40

REMOVE_PAGE_NUMBER_LINES = True
FIX_HYPHENATION = True
JOIN_WRAPPED_LINES = True
MIN_LINE_LEN = 2

# =========================
# OUTILS
# =========================

_page_number_patterns = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*page\s*\d+\s*$", re.I),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*of\s*\d+\s*$", re.I),
    re.compile(r"^\s*page\s*\d+\s*of\s*\d+\s*$", re.I),
]

def looks_like_page_number(line: str) -> bool:
    if not REMOVE_PAGE_NUMBER_LINES:
        return False
    s = line.strip()
    return any(p.match(s) for p in _page_number_patterns)

def normalize_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def cleanup_block_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================
# EXTRACTION
# =========================

def extract_text_clean(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    pages_lines = []
    normalized_counts = Counter()

    for i in range(n_pages):
        page = doc.load_page(i)
        rect = page.rect

        clip = fitz.Rect(
            rect.x0,
            rect.y0 + rect.height * TOP_MARGIN_RATIO,
            rect.x1,
            rect.y1 - rect.height * BOTTOM_MARGIN_RATIO
        )

        raw = page.get_text("text", clip=clip)
        lines = [normalize_line(ln) for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln and len(ln) >= MIN_LINE_LEN]

        for ln in set(lines):
            normalized_counts[ln] += 1

        pages_lines.append(lines)

    repeated_threshold = max(2, int(n_pages * REPEAT_LINE_MIN_PAGES_RATIO))
    repeated_lines = {ln for ln, c in normalized_counts.items() if c >= repeated_threshold}

    cleaned_pages = []
    for lines in pages_lines:
        out = []
        for ln in lines:
            if ln in repeated_lines:
                continue
            if looks_like_page_number(ln):
                continue
            out.append(ln)
        cleaned_pages.append(out)

    text = "\n\n".join("\n".join(lines) for lines in cleaned_pages).strip()

    if FIX_HYPHENATION:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if JOIN_WRAPPED_LINES:
        def join_paragraph(p: str) -> str:
            lines = p.split("\n")
            out = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                if not out:
                    out.append(ln)
                    continue
                prev = out[-1]
                if (not re.search(r"[.!?:;]$", prev)) and re.match(r"^[a-zàâçéèêëîïôûùüÿñæœ]", ln):
                    out[-1] = prev + " " + ln
                else:
                    out.append(ln)
            return "\n".join(out)

        parts = text.split("\n\n")
        parts = [join_paragraph(p) for p in parts]
        text = "\n\n".join(parts)

    return cleanup_block_text(text)

# =========================
# TRAITEMENT DE TOUS LES PDF
# =========================

def process_all_pdfs():
    PDF_DIR_OUT.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Aucun PDF trouvé dans le dossier.")
        return

    for pdf_path in pdf_files:
        print(f"Traitement : {pdf_path.name}")

        try:
            text = extract_text_clean(pdf_path)

            if len(text.strip()) < 50:
                text = (
                    "ATTENTION : très peu de texte extrait.\n"
                    "Le PDF est peut-être scanné (image). OCR nécessaire.\n\n"
                ) + text

            output_file = PDF_DIR_OUT / (pdf_path.stem + ".txt")
            output_file.write_text(text, encoding="utf-8")

            print(f"→ OK : {output_file.name}")

        except Exception as e:
            print(f"Erreur avec {pdf_path.name} : {e}")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    process_all_pdfs()