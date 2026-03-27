# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import re
import fitz  # PyMuPDF

PDF_DIR = Path(r"C:/PYTHON/.entree/Sources")
TXT_DIR_OUT = Path(r"C:/PYTHON/.data/ResultatsPDF")

# Ajustement léger : le programme 1 reste prudent sur les blocs ambigus.
PRESERVE_SENTENCE_LIKE_NUMERIC_BLOCKS = True

# ===============================
# LOG
# ===============================

def log(msg):
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")


# ===============================
# OUTILS TEXTE
# ===============================

def normalize_spaces(text):
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\xad", "")   # soft hyphen
    text = text.replace("￾", "")
    return text

def normalize_line(text):
    text = normalize_spaces(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_like_all_caps_title(text):
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 6:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / max(1, len(letters)) >= 0.85

def has_sentence_shape(text):
    return bool(re.search(r"[a-z]{3,}\s+[a-z]{3,}", text))

def remove_trailing_page_number(text):
    return re.sub(r"\s+\d{1,4}\s*$", "", text).strip()

def remove_page_markers(text):
    text = re.sub(r"<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>", " ", text, flags=re.I)
    text = re.sub(r"<IMAGE FOR PAGE:\s*\d+\s*/\s*\d+>", " ", text, flags=re.I)
    return normalize_line(text)


# ===============================
# FILTRES LIGNES
# ===============================

def is_numeric_heavy_text(text):
    tokens = re.findall(r"\b\w+[\w.%/-]*\b", text)
    if len(tokens) < 6:
        return False
    numeric = sum(1 for t in tokens if re.search(r"\d", t))
    short = sum(1 for t in tokens if len(t) <= 3)
    return (numeric / max(1, len(tokens)) >= 0.45) or (numeric + short >= max(8, int(len(tokens) * 0.75)))

def is_chart_or_table_block(text):
    t = normalize_line(text)
    low = t.lower()

    if is_figure_caption(t) or is_table_caption(t):
        return False

    if low.startswith("source:") or low.startswith("note:"):
        return False

    if len(t) <= 35:
        return False

    # Ajustement prudent : si le bloc ressemble à une vraie phrase,
    # on préfère le laisser au programme 2 plutôt que de le supprimer ici.
    if PRESERVE_SENTENCE_LIKE_NUMERIC_BLOCKS and has_sentence_shape(t):
        numeric_ratio = is_numeric_heavy_text(t)
        if numeric_ratio and len(t.split()) >= 12:
            return False

    keywords = [
        "primary emotion", "secondary emotion", "don't know",
        "net business leader expectation", "net entry-level worker expectation",
        "50 percent or less", "51-75 percent", "76-99 percent", "100 percent"
    ]
    if any(k in low for k in keywords):
        return True

    words = t.split()
    if len(words) >= 8 and is_numeric_heavy_text(t) and not has_sentence_shape(t):
        return True

    if len(words) >= 10:
        title_like = sum(1 for w in words if re.match(r"^[A-Z][a-z]+$", w))
        if title_like / max(1, len(words)) >= 0.75 and not has_sentence_shape(t):
            return True

    return False

def is_probable_title_fragment(text):
    t = normalize_line(text)
    if not t or len(t) > 120:
        return False
    if re.match(r"^\d{1,2}$", t):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 ,&:/()'’\-]{3,100}$", t) and not re.search(r"[.!?]$", t):
        return len(t.split()) <= 14
    return False

def group_blocks_by_band(blocks, y_gap=26):
    if not blocks:
        return []

    ordered = sorted(blocks, key=lambda b: (b[1], b[0]))
    bands = [[ordered[0]]]

    for b in ordered[1:]:
        prev_band = bands[-1]
        last_y = max(x[1] for x in prev_band)
        if abs(b[1] - last_y) <= y_gap:
            prev_band.append(b)
        else:
            bands.append([b])

    return bands


FRONT_MATTER_LINE_PATTERNS = [
    r"^aalborg universitet$",
    r"^downloaded from\b",
    r"^general rights$",
    r"^take down policy$",
    r"^published in\b",
    r"^document version\b",
    r"^publication date\b",
    r"^link to publication\b",
    r"^citation for published version\b",
    r"^please cite the published version\b",
    r"^terms of use\b",
    r"^publisher'?s pdf\b",
    r"^accepted author manuscript\b",
    r"^peer reviewed version\b",
    r"^corresponding author\b",
    r"^author accepted manuscript\b",
]

MID_DOC_METADATA_PATTERNS = [
    r"^doi\b",
    r"^https?://",
    r"^www\.",
    r"^orcid\b",
    r"^conflict of interest",
    r"^data availability statement\b",
    r"^open access\b",
    r"^funding information\b",
]

END_SECTION_HEADERS = {
    "references",
    "bibliography",
    "endnotes",
    "notes",
    "acknowledgements",
    "acknowledgments",
    "author biographies",
    "biographies",
    "appendix",
    "appendices",
    "about the authors",
}

SECTION_HEADERS = {
    "abstract",
    "summary",
    "executive summary",
    "introduction",
    "background",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "implications",
    "future research",
    "foreword",
    "preface",
    "keywords",
    "key words",
}

CASE_STUDY_PATTERNS = [
    r"^in practice\s*:",
    r"^box\s+\d+",
    r"^case study\s*:",
    r"^example\s*:",
    r"^sidebar\b",
    r"^exhibit sidebar\b",
]

def is_front_matter_line(text):
    t = text.lower().strip()
    return any(re.match(p, t, re.I) for p in FRONT_MATTER_LINE_PATTERNS)

def is_metadata_line(text):
    t = text.lower().strip()

    if any(re.match(p, t, re.I) for p in MID_DOC_METADATA_PATTERNS):
        return True

    generic_contains = [
        "all rights reserved",
        "licensed under",
        "creative commons",
        "wiley online library",
        "retrieved from",
        "email:",
        "e-mail:",
        "tel.:",
        "fax:",
    ]
    if any(x in t for x in generic_contains):
        return True

    return False

def is_reference_line(text):
    t = text.strip()

    if re.match(r"^\[\d+\]", t):
        return True

    if re.match(r"^\d+\.\s", t) and len(t) > 80 and ("doi" in t.lower() or "http" in t.lower()):
        return True

    if re.match(r"^[A-Z][A-Za-z'`\-]+,\s+[A-Z]\.", t):
        return True

    return False

def is_equation(text):
    return bool(re.search(r"[=<>±∑∫]", text))

def is_table_complex(text):
    low = text.lower()
    keywords = [
        "loading", "coefficient", "regression", "std.", "p-value",
        "confidence interval", "cronbach", "alpha", "r-squared", "anova"
    ]
    return any(k in low for k in keywords)

def is_short_noise(text):
    t = text.strip()
    if len(t) <= 2:
        return True
    if re.match(r"^[()–—•·,\-:;_/\\]+$", t):
        return True
    if len(t) < 35 and looks_like_all_caps_title(t):
        return True
    return False

def is_end_section_header(text):
    return text.lower().strip() in END_SECTION_HEADERS

def is_section_header(text):
    t = text.strip()

    if t.lower() in SECTION_HEADERS:
        return True

    if re.match(r"^\d+(\.\d+){0,3}\s+[A-Z]", t):
        return True

    if re.match(r"^[A-Z][A-Za-z \-]{2,80}:?$", t) and len(t.split()) <= 8:
        if looks_like_all_caps_title(t) or t.istitle():
            return True

    return False

def is_keywords_line(text):
    return bool(re.match(r"^(keywords|key words)\s*[:\-]", text.strip(), re.I))

def is_bullet_line(text):
    return bool(re.match(r"^[•●▪◦\-–]\s+", text.strip()))

def is_figure_caption(text):
    return bool(re.match(r"^(figure|fig\.)\s*\d*[\s:.\-]", text.strip(), re.I))

def is_table_caption(text):
    return bool(re.match(r"^table\s*\d*[\s:.\-]?", text.strip(), re.I))

def should_force_paragraph_break(text):
    t = text.strip()
    if not t:
        return True
    if t == "ABSTRACT":
        return True
    if t.startswith("[CASE STUDY]") or t.startswith("[/CASE STUDY]"):
        return True
    if t.startswith("FIGURE:") or t.startswith("TABLE:"):
        return True
    if is_section_header(t):
        return True
    if is_keywords_line(t):
        return True
    if is_bullet_line(t):
        return True
    return False


# ===============================
# TRI COLONNE-AWARE
# ===============================

def sort_blocks_column_aware(blocks, page_width):
    if not blocks:
        return blocks

    FULL_WIDTH_RATIO = 0.72
    bands = group_blocks_by_band(blocks)
    result = []

    for band in bands:
        band = sorted(band, key=lambda b: (b[0], b[1]))

        full_width = []
        columns = []
        for b in band:
            width = b[2] - b[0]
            if width >= page_width * FULL_WIDTH_RATIO:
                full_width.append(b)
            else:
                columns.append(b)

        full_width.sort(key=lambda b: (b[1], b[0]))
        columns.sort(key=lambda b: (b[0], b[1]))

        if full_width and columns:
            min_full_y = min(b[1] for b in full_width)
            before_full = [b for b in columns if b[1] < min_full_y]
            after_full = [b for b in columns if b[1] >= min_full_y]
            result.extend(before_full)
            result.extend(full_width)
            result.extend(after_full)
        else:
            result.extend(full_width or columns)

    return result


# ===============================
# DÉTECTION EN-TÊTES/PIEDS RÉPÉTITIFS
# ===============================

def detect_running_headers(doc):
    candidates = {}
    n_pages = len(doc)

    for page in doc:
        page_height = page.rect.height
        for b in page.get_text("blocks"):
            y0, y1, text = b[1], b[3], normalize_line(b[4])
            if not text:
                continue

            if y0 < page_height * 0.07 or y1 > page_height * 0.93:
                normalized = remove_trailing_page_number(text)
                normalized = remove_page_markers(normalized)
                if len(normalized) > 5:
                    candidates[normalized] = candidates.get(normalized, 0) + 1

    threshold = max(2, round(n_pages * 0.30))
    return {t for t, count in candidates.items() if count >= threshold}


# ===============================
# DÉTECTION DU VRAI DÉBUT
# ===============================

def page_text_for_detection(page):
    txt = page.get_text("text")
    txt = normalize_spaces(txt)
    txt = txt.replace("\r", "\n")
    return txt

def score_page_as_content_start(page_text):
    txt = page_text.lower()

    score = 0

    positive_patterns = [
        r"\babstract\b",
        r"\bintroduction\b",
        r"\bexecutive summary\b",
        r"\bsummary\b",
        r"\bkeywords\b",
        r"\bkey words\b",
        r"\b1\.\s+[a-z]",
        r"\b1\s+[a-z]",
    ]
    for p in positive_patterns:
        if re.search(p, txt, re.I):
            score += 2

    negative_patterns = [
        r"\bcontents\b",
        r"\btable of contents\b",
        r"\bdownloaded from\b",
        r"\bgeneral rights\b",
        r"\bpublished in\b",
        r"\bdocument version\b",
        r"\bcitation for published version\b",
        r"\btake down policy\b",
        r"\baccepted author manuscript\b",
    ]
    for p in negative_patterns:
        if re.search(p, txt, re.I):
            score -= 3

    lines = [normalize_line(x) for x in page_text.splitlines() if normalize_line(x)]
    long_lines = sum(1 for x in lines if len(x) >= 70 and has_sentence_shape(x))
    if long_lines >= 3:
        score += 2

    return score

def find_front_matter_end(doc):
    max_scan = min(len(doc), 8)

    best_page = 0
    best_score = -999

    for page_num in range(max_scan):
        page = doc[page_num]
        txt = page_text_for_detection(page)
        score = score_page_as_content_start(txt)

        if score > best_score:
            best_score = score
            best_page = page_num

        # bon signal : vrai contenu trouvé
        if score >= 3:
            return page_num

    # sécurité : si rien de clair, ne saute pas trop de pages
    if best_score >= 1:
        return best_page

    return 0


# ===============================
# EXTRACTION DES BLOCS
# ===============================

def block_to_text(block_text):
    text = normalize_spaces(block_text)
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)   # mot coupé en fin de ligne
    text = re.sub(r"\s*\n\s*", " ", text)             # lignes internes -> espace
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_page_markers(text)
    return text

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)

    running_headers = detect_running_headers(doc)
    if running_headers:
        log(
            f"En-têtes/pieds détectés ({len(running_headers)}) : "
            f"{list(running_headers)[:3]}{'...' if len(running_headers) > 3 else ''}"
        )

    first_content_page = find_front_matter_end(doc)
    if first_content_page > 0:
        log(f"Front matter détecté : {first_content_page} page(s) ignorée(s).")

    full_text = []

    for page_num, page in enumerate(doc):
        if page_num < first_content_page:
            continue

        page_width = page.rect.width
        blocks = page.get_text("blocks")

        filtered = []
        for b in blocks:
            raw = b[4].strip()
            if not raw:
                continue

            text = block_to_text(raw)
            if not text:
                continue

            normalized = remove_trailing_page_number(text)
            if normalized in running_headers:
                continue

            if is_short_noise(normalized):
                continue

            if is_chart_or_table_block(text):
                continue

            filtered.append((b[0], b[1], b[2], b[3], text, *b[5:]))

        sorted_blocks = sort_blocks_column_aware(filtered, page_width)

        for b in sorted_blocks:
            text = b[4].strip()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


# ===============================
# COUPE DÉBUT DE TEXTE
# ===============================

def trim_leading_noise(lines):
    """
    Supprime les lignes parasites au début du texte jusqu'au premier vrai signal de contenu.
    """
    start_idx = 0

    for i, line in enumerate(lines[:120]):
        l = line.strip()
        low = l.lower()

        good_start = (
            low == "abstract" or
            low == "introduction" or
            low == "executive summary" or
            is_keywords_line(l) or
            re.match(r"^\d+(\.\d+)?\s+[A-Z]", l)
        )

        bad_start = (
            is_front_matter_line(l) or
            is_metadata_line(l) or
            low in {"contents", "table of contents"} or
            "downloaded from" in low or
            "published in" in low
        )

        if good_start:
            start_idx = i
            break

        if not bad_start and len(l) > 80 and has_sentence_shape(l):
            start_idx = i
            break

    return lines[start_idx:]


# ===============================
# NETTOYAGE PRINCIPAL
# ===============================

def clean_text(text):
    lines = [normalize_line(x) for x in text.split("\n")]
    lines = [x for x in lines if x]
    lines = trim_leading_noise(lines)

    cleaned = []
    seen_abstract = False
    consecutive_ref_lines = 0

    for line in lines:
        l = line.strip()
        low = l.lower()

        if not l:
            continue

        if re.match(r"^\d{1,4}$", l):
            continue

        if is_probable_title_fragment(l) and cleaned:
            prev = cleaned[-1]
            if is_probable_title_fragment(prev) and len(prev) + len(l) <= 150:
                cleaned[-1] = f"{prev} {l}"
                continue

        if is_front_matter_line(l):
            continue

        if is_metadata_line(l):
            continue

        # Ajustement : on laisse davantage le nettoyage sémantique au programme 2.
        # On ne supprime plus ici les lignes d'équations / tableaux complexes.

        if is_short_noise(l):
            continue

        # fin de document
        if is_end_section_header(low):
            if len(cleaned) > 25:
                log(f"Fin de document détectée sur section terminale : {l}")
                break
            continue

        if is_reference_line(l):
            consecutive_ref_lines += 1
            if consecutive_ref_lines >= 4 and len(cleaned) > 25:
                log("Fin de document détectée : bloc de références.")
                break
            continue
        else:
            consecutive_ref_lines = 0

        # normalisation abstract
        if re.match(r"^(abstract|summary)\s*$", l, re.I):
            if seen_abstract:
                continue
            seen_abstract = True
            cleaned.append("ABSTRACT")
            continue

        # normalisation keywords
        if is_keywords_line(l):
            cleaned.append(re.sub(r"^(key words|keywords)\s*[:\-]\s*", "KEYWORDS: ", l, flags=re.I))
            continue

        if is_table_caption(l):
            cleaned.append(f"TABLE: {l}")
            continue

        if is_figure_caption(l):
            if len(l) > 20:
                cleaned.append(f"FIGURE: {l}")
            continue

        # titres trop longs en capitales -> souvent bruit
        if looks_like_all_caps_title(l) and len(l) > 80:
            continue

        cleaned.append(l)

    return "\n".join(cleaned)


# ===============================
# BALISAGE DES ÉTUDES DE CAS
# ===============================

def tag_case_studies(text):
    lines = text.split("\n")
    result = []
    in_case_study = False

    for line in lines:
        l = line.strip()
        if not l:
            continue

        is_cs_start = any(re.match(p, l, re.I) for p in CASE_STUDY_PATTERNS)

        if in_case_study and (is_cs_start or is_section_header(l)):
            result.append("[/CASE STUDY]")
            in_case_study = False

        if is_cs_start:
            result.append("[CASE STUDY]")
            in_case_study = True

        result.append(l)

    if in_case_study:
        result.append("[/CASE STUDY]")

    return "\n".join(result)


# ===============================
# RECONSTRUCTION DES PARAGRAPHES
# ===============================

def reconstruct_paragraphs(text):
    lines = text.split("\n")
    result = []
    buffer = []

    def flush():
        if buffer:
            para = " ".join(buffer)
            para = re.sub(r"\s+", " ", para).strip()
            if para:
                result.append(para)
            buffer.clear()

    for line in lines:
        l = line.strip()

        if not l:
            flush()
            continue

        if should_force_paragraph_break(l):
            flush()
            result.append(l)
            continue

        # une ligne courte sans ponctuation finale ressemble souvent à un titre
        if len(l) <= 90 and not re.search(r"[.!?;:]\s*$", l) and (
            looks_like_all_caps_title(l) or re.match(r"^[A-Z][A-Za-z0-9 ,\-()]{3,80}$", l)
        ):
            if re.match(r"^\d{1,2}$", l):
                continue
            flush()
            result.append(l)
            continue

        buffer.append(l)

        # fin probable de paragraphe
        if re.search(r'[.!?]"?\s*$', l):
            flush()

    flush()
    return "\n\n".join(result)


# ===============================
# FINAL CLEAN
# ===============================

def normalize(text):
    text = normalize_spaces(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


# ===============================
# PIPELINE
# ===============================

def process_pdf(pdf_path):
    log(f"Processing {pdf_path.name}")

    try:
        raw = extract_text_blocks(pdf_path)
        cleaned = clean_text(raw)
        tagged = tag_case_studies(cleaned)
        paragraphs = reconstruct_paragraphs(tagged)
        final = normalize(paragraphs)

        TXT_DIR_OUT.mkdir(parents=True, exist_ok=True)
        out_path = TXT_DIR_OUT / f"{pdf_path.stem}.txt"
        out_path.write_text(final, encoding="utf-8")

        log(f"Saved -> {out_path}")

    except fitz.fitz.FitzError as e:
        log(f"ERREUR PDF corrompu ou protégé ({pdf_path.name}) : {e}")
    except PermissionError as e:
        log(f"ERREUR permission refusée ({pdf_path.name}) : {e}")
    except Exception as e:
        log(f"ERREUR inattendue ({pdf_path.name}) : {type(e).__name__} : {e}")


# ===============================
# MAIN
# ===============================

def main():
    log("START")

    pdfs = list(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        log("Aucun fichier PDF trouvé dans le répertoire source.")
        return

    log(f"{len(pdfs)} fichier(s) PDF trouvé(s).")

    errors = 0
    for pdf in pdfs:
        try:
            process_pdf(pdf)
        except Exception:
            errors += 1

    log(f"DONE - {len(pdfs) - errors}/{len(pdfs)} fichier(s) traité(s) avec succès.")


if __name__ == "__main__":
    main()