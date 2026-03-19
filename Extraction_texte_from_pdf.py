"""
Extraction "propre" de texte depuis un PDF (sans numéros de page, en-têtes/pieds répétitifs)
- Fonctionne bien pour les PDF "texte" (pas scannés).
- Si le PDF est scanné (images), il faudra un OCR (non inclus ici).

Dépendance :
    pip install pymupdf
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from collections import Counter

# =========================
# PARAMÈTRES (à ajuster)
# =========================

INPUT_PDF = "C:/PYTHON/.entree/ETP_crohn.pdf"
OUTPUT_TXT = "C:/PYTHON/.data/ETPcrohn.txt"

# On ignore le haut/bas de page (souvent en-têtes/pieds)
TOP_MARGIN_RATIO = 0.10      # 0.08 à 0.15 recommandé
BOTTOM_MARGIN_RATIO = 0.10   # 0.08 à 0.15 recommandé

# Filtrage de lignes répétées entre pages (en-têtes/pieds typiques)
REPEAT_LINE_MIN_PAGES_RATIO = 0.40  # si une ligne apparaît sur >= 40% des pages => on la supprime

# Filtrage des numéros de page (et variantes)
REMOVE_PAGE_NUMBER_LINES = True

# Nettoyage typographique
FIX_HYPHENATION = True       # recolle "trans-\nform" => "transform"
JOIN_WRAPPED_LINES = True    # recolle les lignes coupées (mise en page)
MIN_LINE_LEN = 2             # ignore les lignes très courtes (bruit)


# =========================
# OUTILS DE NETTOYAGE
# =========================

_page_number_patterns = [
    re.compile(r"^\s*\d+\s*$"),                              # "12"
    re.compile(r"^\s*page\s*\d+\s*$", re.I),                 # "Page 12"
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),                    # "12/48"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),                      # "- 12 -"
    re.compile(r"^\s*\d+\s*of\s*\d+\s*$", re.I),             # "12 of 48"
    re.compile(r"^\s*page\s*\d+\s*of\s*\d+\s*$", re.I),      # "Page 12 of 48"
]

def looks_like_page_number(line: str) -> bool:
    if not REMOVE_PAGE_NUMBER_LINES:
        return False
    s = line.strip()
    for p in _page_number_patterns:
        if p.match(s):
            return True
    return False

def normalize_line(line: str) -> str:
    # Normalisation légère pour comparer les répétitions
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def cleanup_block_text(text: str) -> str:
    # Nettoyage basique des espaces
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =========================
# EXTRACTION PRINCIPALE
# =========================

def extract_text_clean(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    # 1) Extraction brute par page en ignorant haut/bas de page
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

        # get_text("text") donne du texte en lignes. clip retire marges.
        raw = page.get_text("text", clip=clip)

        # Split en lignes, nettoyage minimal
        lines = [normalize_line(ln) for ln in raw.splitlines()]
        # Retire lignes vides / trop courtes
        lines = [ln for ln in lines if ln and len(ln) >= MIN_LINE_LEN]

        # Compte des lignes normalisées pour repérer les répétitions
        for ln in set(lines):
            normalized_counts[ln] += 1

        pages_lines.append(lines)

    # 2) Détecter les lignes répétées sur beaucoup de pages (en-têtes/pieds typiques)
    repeated_threshold = max(2, int(n_pages * REPEAT_LINE_MIN_PAGES_RATIO))
    repeated_lines = {ln for ln, c in normalized_counts.items() if c >= repeated_threshold}

    # 3) Filtrer lignes répétées + numéros de page + bruit
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

    # 4) Reconstituer le texte (avec corrections optionnelles)
    # On sépare les pages par une ligne vide pour garder un minimum de structure.
    text = "\n\n".join("\n".join(lines) for lines in cleaned_pages).strip()

    if FIX_HYPHENATION:
        # "trans-\nform" => "transform"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if JOIN_WRAPPED_LINES:
        # Heuristique : recoller les sauts de ligne "mise en page" au sein d'un paragraphe.
        # On garde les doubles sauts de ligne (paragraphes).
        # On recolle si la ligne ne finit pas par ponctuation forte.
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
                # si la ligne précédente ne termine pas par . ! ? : ; et que la nouvelle commence par minuscule
                if (not re.search(r"[.!?:;]$", prev)) and re.match(r"^[a-zàâçéèêëîïôûùüÿñæœ]", ln):
                    out[-1] = prev + " " + ln
                else:
                    out.append(ln)
            return "\n".join(out)

        parts = text.split("\n\n")
        parts = [join_paragraph(p) for p in parts]
        text = "\n\n".join(parts)

    text = cleanup_block_text(text)
    return text


def main():
    text = extract_text_clean(INPUT_PDF)

    # Si quasiment rien n'est extrait, le PDF est probablement scanné (image)
    if len(text.strip()) < 50:
        warning = (
            "ATTENTION : très peu de texte extrait.\n"
            "Le PDF est peut-être scanné (image). Dans ce cas, il faut un OCR.\n\n"
        )
        text = warning + text

    Path(OUTPUT_TXT).write_text(text, encoding="utf-8")
    print("Texte extrait écrit dans :", OUTPUT_TXT)
    print("Taille (caractères) :", len(text))


if __name__ == "__main__":
    main()
