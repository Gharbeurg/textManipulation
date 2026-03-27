# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import unicodedata

# ============================================================
# CONFIGURATION
# ============================================================

TXT_DIR_IN = Path(r"C:/PYTHON/.data/ResultatsPDF")
TXT_DIR_OUT = Path(r"C:/PYTHON/.data/ResultatsTXT_clean")

MIN_LINE_LEN_TO_KEEP = 3
MIN_REPEAT_COUNT_FOR_GLOBAL_NOISE = 3
AGGRESSIVE_END_CUT = True

# Nouveau : gestion du début
REMOVE_FOREWORD = True
REMOVE_EXECUTIVE_SUMMARY_IN_FRONT = False   # laisse False
MAX_FRONT_SCAN_LINES = 260

# Si vrai, on coupe fort dès qu'on détecte un sommaire en tête
AGGRESSIVE_TOC_REMOVAL = True

# Protection légère des vraies sections utiles
PROTECTED_START_HEADERS = {
    "abstract", "executive summary", "introduction", "summary",
    "overview", "keywords", "key words"
}

PROTECTED_END_HEADERS = {
    "conclusion", "conclusions", "discussion", "final thoughts",
    "practical applications", "theoretical applications", "implications"
}

# ============================================================
# LOG
# ============================================================

def log(msg):
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")


# ============================================================
# OUTILS TEXTE
# ============================================================

def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\xad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("￾", "")
    text = text.replace("\t", " ")
    return text

def normalize_line(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def canonical_line(text: str) -> str:
    text = normalize_line(text).lower()
    text = strip_accents(text)
    text = re.sub(r"\b\d{1,4}\b", " ", text)
    text = re.sub(r"[_|•·■►◦▪]+", " ", text)
    text = re.sub(r"[^\w\s:/.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_mostly_upper(text: str) -> bool:
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", text)
    if len(letters) < 6:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / max(1, len(letters)) >= 0.80

def looks_like_sentence(text: str) -> bool:
    return bool(re.search(r"[a-zà-ÿ]{3,}\s+[a-zà-ÿ]{3,}", text, re.I))

def ends_like_paragraph(text: str) -> bool:
    return bool(re.search(r'[.!?…]"?\)?\s*$', text))

def remove_lonely_page_number(line: str) -> str:
    if re.match(r"^\s*\d{1,4}\s*$", line):
        return ""
    return line

def remove_page_like_suffix(line: str) -> str:
    return re.sub(r"\s+\d{1,4}\s*$", "", line).strip()

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def starts_like_mid_sentence(line: str) -> bool:
    l = normalize_line(line)
    if not l:
        return False
    if re.match(r"^[a-zà-ÿ]", l):
        return True
    if re.match(r"^[,.;:)]", l):
        return True
    return False

def is_numeric_heavy_line(line: str) -> bool:
    l = normalize_line(line)
    tokens = re.findall(r"\b\w+[\w.%/-]*\b", l)
    if len(tokens) < 4:
        return False
    numeric = sum(1 for t in tokens if re.search(r"\d", t))
    very_short = sum(1 for t in tokens if len(t) <= 3)
    return numeric / max(1, len(tokens)) >= 0.5 or (numeric + very_short) / max(1, len(tokens)) >= 0.8

def is_chart_data_line(line: str) -> bool:
    l = normalize_line(line)
    low = l.lower()

    if not l:
        return False

    if low.startswith(("figure", "table", "source:", "note:")):
        return False

    if re.fullmatch(r"\d+%?(\s+\d+%?){1,12}", l):
        return True

    keywords = [
        "primary emotion", "secondary emotion", "don't know",
        "50 percent or less", "51-75 percent", "76-99 percent", "100 percent",
        "ai agents", "generative ai"
    ]
    if any(k in low for k in keywords) and is_numeric_heavy_line(l):
        return True

    words = l.split()
    if len(words) >= 6 and is_numeric_heavy_line(l):
        return True

    if len(words) >= 8:
        cap_words = sum(1 for w in words if re.match(r"^[A-Z][a-z]+$", w))
        if cap_words / max(1, len(words)) >= 0.75 and not looks_like_sentence(l):
            return True

    return False

def remove_chart_data_runs(lines: list[str]) -> list[str]:
    result = []
    i = 0
    n = len(lines)

    while i < n:
        if is_chart_data_line(lines[i]):
            j = i
            hits = 0
            while j < n and (is_chart_data_line(lines[j]) or len(normalize_line(lines[j])) <= 3):
                if is_chart_data_line(lines[j]):
                    hits += 1
                j += 1

            if hits >= 3:
                i = j
                continue

        result.append(lines[i])
        i += 1

    return result

def merge_broken_starts(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    start = 0
    while start < min(12, len(lines)) and starts_like_mid_sentence(lines[start]):
        start += 1

    lines = lines[start:] if start else lines
    if not lines:
        return lines

    result = [lines[0]]
    for line in lines[1:]:
        l = normalize_line(line)
        if starts_like_mid_sentence(l) and result:
            prev = result[-1]
            if len(prev) < 220 and not ends_like_paragraph(prev):
                result[-1] = f"{prev} {l}"
                continue
        result.append(l)

    return result

# ============================================================
# MOTIFS DE BRUIT
# ============================================================

FRONT_NOISE_PATTERNS = [
    r"^contents$",
    r"^table of contents$",
    r"^list of figures$",
    r"^list of tables$",
    r"^contributors$",
    r"^appendix$",
    r"^appendices$",
    r"^endnotes$",
    r"^about the authors$",
    r"^author biographies$",
    r"^biographies$",
]

ACADEMIC_METADATA_PATTERNS = [
    r"^published in\b",
    r"^document version\b",
    r"^publication date\b",
    r"^link to publication\b",
    r"^citation for published version\b",
    r"^please cite the published version\b",
    r"^downloaded from\b",
    r"^general rights$",
    r"^take down policy$",
    r"^accepted author manuscript\b",
    r"^publisher'?s pdf\b",
    r"^peer reviewed version\b",
    r"^corresponding author\b",
    r"^author accepted manuscript\b",
    r"^doi\b",
    r"^orcid\b",
    r"^open access\b",
    r"^data availability statement\b",
    r"^funding information\b",
    r"^supplementary material\b",
]

GENERIC_NOISE_CONTAINS = [
    "all rights reserved",
    "creative commons",
    "wiley online library",
    "downloaded from",
    "published by",
    "citation for published version",
    "correspondence:",
    "corresponding author",
    "email:",
    "e-mail:",
    "tel.:",
    "fax:",
    "http://",
    "https://",
    "www.",
]

END_SECTION_HEADERS = {
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "endnotes",
    "notes",
    "acknowledgements",
    "acknowledgments",
    "author biographies",
    "biographies",
    "about the authors",
    "works cited",
    "contributors",
}

MAIN_START_HEADERS = {
    "abstract",
    "introduction",
    "executive summary",
    "summary",
    "overview",
    "chapter 1",
}

STRONG_START_PATTERNS = [
    r"^abstract$",
    r"^introduction$",
    r"^executive summary$",
    r"^summary$",
    r"^(keywords|key words)\s*[:\-]",
    r"^\d+(\.\d+){0,3}\s+[A-Z]",
    r"^chapter\s+\d+\b",
]

CASE_STUDY_PATTERNS = [
    r"^in practice\s*:",
    r"^box\s+\d+",
    r"^case study\s*:",
    r"^example\s*:",
    r"^at a glance$",
]

# ============================================================
# TESTS DE LIGNES
# ============================================================

def is_academic_metadata_line(line: str) -> bool:
    low = normalize_line(line).lower()
    if any(re.match(p, low, re.I) for p in ACADEMIC_METADATA_PATTERNS):
        return True
    if any(x in low for x in GENERIC_NOISE_CONTAINS):
        return True
    return False

def is_front_noise_line(line: str) -> bool:
    low = normalize_line(line).lower()
    if any(re.match(p, low, re.I) for p in FRONT_NOISE_PATTERNS):
        return True
    return False

def is_end_section_header(line: str) -> bool:
    return normalize_line(line).lower() in END_SECTION_HEADERS

def is_protected_start_header(line: str) -> bool:
    return normalize_line(line).lower() in PROTECTED_START_HEADERS

def is_protected_end_header(line: str) -> bool:
    return normalize_line(line).lower() in PROTECTED_END_HEADERS

def is_strong_start_line(line: str) -> bool:
    l = normalize_line(line)
    low = l.lower()
    if low in MAIN_START_HEADERS:
        return True
    if any(re.match(p, l, re.I) for p in STRONG_START_PATTERNS):
        return True
    return False

def is_bullet_line(line: str) -> bool:
    l = line.strip()
    return bool(re.match(r"^[•●▪◦►■\-–]\s+", l))

def is_figure_caption(line: str) -> bool:
    l = line.strip()
    return bool(re.match(r"^(figure|fig\.)\s*\d*[\s:.\-]", l, re.I))

def is_table_caption(line: str) -> bool:
    l = line.strip()
    return bool(re.match(r"^table\s*\d*[\s:.\-]", l, re.I))

def is_probable_reference_line(line: str) -> bool:
    l = normalize_line(line)

    if re.match(r"^\[\d+\]", l):
        return True

    if re.match(r"^\d+\.\s", l) and ("doi" in l.lower() or "http" in l.lower()):
        return True

    if re.match(r"^[A-Z][A-Za-z'`\-]+,\s+[A-Z]\.", l):
        return True

    if re.search(r"\(\d{4}[a-z]?\)", l) and len(l) > 60:
        return True

    return False

def is_short_noise(line: str) -> bool:
    l = normalize_line(line)

    if len(l) < MIN_LINE_LEN_TO_KEEP:
        return True

    if re.match(r"^[\W_]+$", l):
        return True

    if l.lower() in {"page", "source", "figure", "table"}:
        return True

    return False

def is_probable_title(line: str) -> bool:
    l = normalize_line(line)

    if not l:
        return False

    if len(l) > 120:
        return False

    if is_strong_start_line(l):
        return True

    if is_mostly_upper(l):
        return True

    if re.match(r"^\d+(\.\d+){0,3}\s+[A-Z]", l):
        return True

    if re.match(r"^[A-Z][A-Za-z0-9 ,:;()'’/\-]{2,100}$", l) and not ends_like_paragraph(l):
        words = l.split()
        if 1 <= len(words) <= 12:
            return True

    return False

def is_case_study_start(line: str) -> bool:
    l = normalize_line(line)
    return any(re.match(p, l, re.I) for p in CASE_STUDY_PATTERNS)

def is_foreword_line(line: str) -> bool:
    return normalize_line(line).lower() == "foreword"

def is_toc_entry(line: str) -> bool:
    """
    Détecte une ligne de sommaire, ex:
    3 The future of AI and work
    3.1 Four imperatives...
    Conclusion 20
    Appendix: Questions sent 21
    """
    l = normalize_line(line)

    patterns = [
        r"^\d+(\.\d+){0,3}\s+[A-Z].{0,120}$",
        r"^[A-Z][A-Za-z ,&:/\-\(\)']{2,120}\s+\d{1,4}$",
        r"^[A-Z][A-Za-z ,&:/\-\(\)']{2,120}\s+\d{1,4}\s+[A-Z].*$",
    ]

    if any(re.match(p, l) for p in patterns):
        if not ends_like_paragraph(l):
            return True

    keywords = ["appendix", "contributors", "endnotes", "conclusion", "foreword"]
    if any(k in l.lower() for k in keywords):
        if re.search(r"\b\d{1,4}\b", l):
            return True

    return False

def is_long_author_line(line: str) -> bool:
    """
    Ex: 'Erik Brynjolfsson Jerry Yang and Akiko...'
    """
    l = normalize_line(line)
    if len(l) > 180:
        return False
    if looks_like_sentence(l):
        return False
    words = l.split()
    if len(words) < 4:
        return False
    capitalized = sum(1 for w in words if re.match(r"^[A-Z][a-zA-Z'’\-]+$", w))
    return capitalized / max(1, len(words)) >= 0.7


# ============================================================
# LECTURE / PREP
# ============================================================

def read_txt(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_spaces(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [normalize_line(remove_lonely_page_number(x)) for x in text.split("\n")]
    lines = [x for x in lines if x]
    return lines


# ============================================================
# SUPPRESSION DES LIGNES GLOBALEMENT RÉPÉTÉES
# ============================================================

def detect_global_repeated_noise(lines: list[str]) -> set[str]:
    canon_lines = [canonical_line(x) for x in lines]
    counts = Counter(canon_lines)
    total = len(lines)

    noise = set()
    for canon, count in counts.items():
        if not canon:
            continue

        if len(canon) < 6:
            continue

        if count >= MIN_REPEAT_COUNT_FOR_GLOBAL_NOISE and (count / max(1, total)) >= 0.015:
            if canon in {
                "abstract", "introduction", "references", "bibliography",
                "executive summary", "summary", "keywords", "key words",
                "foreword"
            }:
                continue
            noise.add(canon)

    return noise

def remove_global_repeated_noise(lines: list[str]) -> list[str]:
    repeated_noise = detect_global_repeated_noise(lines)
    if not repeated_noise:
        return lines

    cleaned = []
    for line in lines:
        canon = canonical_line(line)
        if canon in repeated_noise and len(line) < 120:
            continue
        cleaned.append(line)

    return cleaned


# ============================================================
# SUPPRESSION DU SOMMAIRE EN TÊTE
# ============================================================

def detect_toc_block(lines: list[str]) -> tuple[int, int] | None:
    """
    Cherche un bloc de sommaire au début du document.
    Retourne (start, end_excluded) si trouvé.
    """
    max_scan = min(len(lines), 80)
    i = 0

    while i < max_scan:
        if is_toc_entry(lines[i]) or normalize_line(lines[i]).lower() in {"contents", "table of contents"}:
            start = i
            j = i
            toc_hits = 0

            while j < max_scan:
                l = lines[j]
                low = l.lower()

                if is_toc_entry(l):
                    toc_hits += 1
                    j += 1
                    continue

                if low in {"contents", "table of contents"}:
                    toc_hits += 1
                    j += 1
                    continue

                if len(l) <= 80 and low in {"foreword", "executive summary", "introduction", "abstract"}:
                    # probable fin du sommaire
                    break

                if len(l) <= 50 and low in {"contributors", "appendix", "endnotes"}:
                    toc_hits += 1
                    j += 1
                    continue

                # ligne de transition : on tolère un peu
                if len(l) <= 25 and not looks_like_sentence(l):
                    j += 1
                    continue

                break

            if toc_hits >= 3:
                return start, j

        i += 1

    return None

def remove_front_toc(lines: list[str]) -> list[str]:
    toc_block = detect_toc_block(lines)
    if not toc_block:
        return lines

    start, end = toc_block
    log(f"Sommaire détecté en tête : lignes {start} à {end}")
    return lines[:start] + lines[end:]


# ============================================================
# COUPE DU DÉBUT
# ============================================================

def score_start_candidate(lines: list[str], i: int) -> int:
    line = normalize_line(lines[i])
    low = line.lower()
    score = 0

    if is_strong_start_line(line):
        score += 12

    if low == "foreword":
        score += 6

    if looks_like_sentence(line) and len(line) >= 70:
        score += 4

    window = lines[i:i+8]
    long_sentences = sum(1 for x in window if len(x) >= 60 and looks_like_sentence(x))
    titles = sum(1 for x in window if is_probable_title(x))

    score += long_sentences
    score += min(titles, 2)

    if is_front_noise_line(line):
        score -= 10

    if is_academic_metadata_line(line):
        score -= 8

    if is_toc_entry(line):
        score -= 12

    if "contents" in low or "table of contents" in low:
        score -= 15

    if "downloaded from" in low or "citation for published version" in low:
        score -= 12

    if is_long_author_line(line):
        score -= 5

    return score

def find_best_start_index(lines: list[str]) -> int:
    max_scan = min(len(lines), MAX_FRONT_SCAN_LINES)

    # 1) priorités fortes
    priority_headers = ["abstract", "executive summary", "introduction"]

    for header in priority_headers:
        for i in range(max_scan):
            low = normalize_line(lines[i]).lower()
            if low == header:
                if header == "executive summary" and REMOVE_EXECUTIVE_SUMMARY_IN_FRONT:
                    continue
                return i

    if REMOVE_FOREWORD:
        # si Foreword est présent, on essaie de sauter jusqu'à Executive summary / Introduction / Abstract
        foreword_idx = None
        for i in range(max_scan):
            if is_foreword_line(lines[i]):
                foreword_idx = i
                break

        if foreword_idx is not None:
            for j in range(foreword_idx + 1, max_scan):
                low = normalize_line(lines[j]).lower()
                if low in {"executive summary", "introduction", "abstract"}:
                    return j

    # 2) scoring général
    best_idx = 0
    best_score = -999

    for i in range(max_scan):
        score = score_start_candidate(lines, i)

        if score > best_score:
            best_score = score
            best_idx = i

        if score >= 14:
            return i

    return best_idx

def trim_front(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    if AGGRESSIVE_TOC_REMOVAL:
        lines = remove_front_toc(lines)

    idx = find_best_start_index(lines)

    while idx < len(lines) and starts_like_mid_sentence(lines[idx]):
        idx += 1

    if idx > 0 and idx < len(lines):
        # Ajustement : si on est déjà sur un vrai header utile,
        # on évite de remonter de 2 lignes et de réintroduire du bruit.
        if not is_protected_start_header(lines[idx]):
            back = max(0, idx - 2)
            if back < idx and not starts_like_mid_sentence(lines[back]) and not is_academic_metadata_line(lines[back]):
                idx = back
        log(f"Début utile détecté à la ligne {idx}")
        return lines[idx:]
    return lines


# ============================================================
# COUPE DE LA FIN
# ============================================================

def trim_end(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    ref_streak = 0
    seen_real_closing_section = False

    for i, line in enumerate(lines):
        low = line.lower()

        if is_protected_end_header(line):
            seen_real_closing_section = True
            continue

        if is_end_section_header(line):
            if i > 20:
                log(f"Fin détectée sur section terminale : {line}")
                return lines[:i]

        if is_probable_reference_line(line):
            ref_streak += 1
            if ref_streak >= 4 and i > 20:
                log("Fin détectée sur bloc de références")
                return lines[:i - 3]
        else:
            ref_streak = 0

        if AGGRESSIVE_END_CUT:
            if low in {"appendix", "appendices", "author biographies", "about the authors", "contributors"} and i > 20:
                # Ajustement : on coupe agressivement seulement si l'on est déjà
                # arrivé dans une vraie zone de fin, ou très loin dans le document.
                if seen_real_closing_section or i >= int(len(lines) * 0.70):
                    log(f"Fin agressive détectée : {line}")
                    return lines[:i]

    return lines


# ============================================================
# FILTRAGE LIGNE PAR LIGNE
# ============================================================

def filter_lines(lines: list[str]) -> list[str]:
    cleaned = []

    for idx, line in enumerate(lines):
        l = normalize_line(line)

        if not l:
            continue

        l = remove_page_like_suffix(l)

        if not l:
            continue

        if is_short_noise(l):
            continue

        if is_academic_metadata_line(l):
            continue

        if REMOVE_FOREWORD and l.lower() == "foreword":
            continue

        # retire encore d'éventuelles lignes de sommaire survivantes au tout début
        if idx < 40 and is_toc_entry(l):
            continue

        # lignes auteur/fonction souvent parasites
        if is_long_author_line(l):
            continue

        # lignes très courtes en capitales = souvent bruit
        if len(l) <= 40 and is_mostly_upper(l) and not is_strong_start_line(l):
            continue

        if is_chart_data_line(l):
            cleaned.append("[FIGURE/TABLE DATA OMITTED]")
            continue

        if is_figure_caption(l):
            if len(l) < 25:
                continue
            cleaned.append(f"FIGURE: {l}")
            continue

        if is_table_caption(l):
            if len(l) < 20:
                continue
            cleaned.append(f"TABLE: {l}")
            continue

        cleaned.append(l)

    return cleaned


# ============================================================
# DÉDOUBLONNAGE LOCAL
# ============================================================

def dedupe_nearby_lines(lines: list[str], window: int = 8) -> list[str]:
    result = []
    recent = []

    for line in lines:
        canon = canonical_line(line)

        if canon and canon in recent:
            continue

        result.append(line)
        recent.append(canon)

        if len(recent) > window:
            recent.pop(0)

    return result


# ============================================================
# NORMALISATION DE STRUCTURE
# ============================================================

def normalize_special_lines(lines: list[str]) -> list[str]:
    result = []
    abstract_seen = False
    keywords_seen = False
    exec_summary_seen = False

    for line in lines:
        l = normalize_line(line)
        low = l.lower()

        if low in {"abstract", "summary"}:
            if abstract_seen:
                continue
            abstract_seen = True
            result.append("ABSTRACT")
            continue

        if low == "executive summary":
            if exec_summary_seen:
                continue
            exec_summary_seen = True
            result.append("Executive summary")
            continue

        if re.match(r"^(keywords|key words)\s*[:\-]\s*", l, re.I):
            if keywords_seen:
                continue
            keywords_seen = True
            l = re.sub(r"^(keywords|key words)\s*[:\-]\s*", "KEYWORDS: ", l, flags=re.I)
            result.append(l)
            continue

        result.append(l)

    return result

def tag_case_studies(lines: list[str]) -> list[str]:
    result = []
    in_case = False

    for line in lines:
        l = normalize_line(line)

        if is_case_study_start(l):
            if in_case:
                result.append("[/CASE STUDY]")
            result.append("[CASE STUDY]")
            result.append(l)
            in_case = True
            continue

        if in_case and is_probable_title(l) and not is_case_study_start(l):
            result.append("[/CASE STUDY]")
            in_case = False

        result.append(l)

    if in_case:
        result.append("[/CASE STUDY]")

    return result


# ============================================================
# RECONSTRUCTION DES PARAGRAPHES
# ============================================================

def should_force_break(line: str) -> bool:
    l = normalize_line(line)

    if not l:
        return True

    if l in {"ABSTRACT", "Executive summary"}:
        return True

    if l.startswith("KEYWORDS:"):
        return True

    if l in {"[CASE STUDY]", "[/CASE STUDY]"}:
        return True

    if l.startswith("FIGURE:") or l.startswith("TABLE:"):
        return True

    if is_bullet_line(l):
        return True

    if is_probable_title(l):
        return True

    return False

def reconstruct_paragraphs(lines: list[str]) -> str:
    out = []
    buffer = []

    def flush():
        if not buffer:
            return
        para = " ".join(buffer)
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            out.append(para)
        buffer.clear()

    for line in lines:
        l = normalize_line(line)

        if not l:
            flush()
            continue

        if should_force_break(l):
            flush()
            out.append(l)
            continue

        if is_bullet_line(l):
            flush()
            out.append(l)
            continue

        if len(l) <= 90 and is_probable_title(l):
            flush()
            out.append(l)
            continue

        buffer.append(l)

        if ends_like_paragraph(l):
            flush()

    flush()
    return "\n\n".join(out)


# ============================================================
# NETTOYAGE FINAL
# ============================================================

def final_cleanup(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)
    return text.strip()


# ============================================================
# TRAITEMENT D'UN FICHIER
# ============================================================

def process_txt_file(path_in: Path, path_out: Path):
    log(f"Traitement : {path_in.name}")

    lines = read_txt(path_in)
    initial_count = len(lines)

    lines = remove_global_repeated_noise(lines)
    after_global_noise = len(lines)

    lines = trim_front(lines)
    after_front = len(lines)

    lines = trim_end(lines)
    after_end = len(lines)

    lines = filter_lines(lines)
    lines = remove_chart_data_runs(lines)
    lines = merge_broken_starts(lines)
    after_filter = len(lines)

    lines = dedupe_nearby_lines(lines)
    after_dedupe = len(lines)

    lines = normalize_special_lines(lines)
    lines = tag_case_studies(lines)

    text = reconstruct_paragraphs(lines)
    text = final_cleanup(text)

    safe_write_text(path_out, text)

    log(
        f"Terminé : {path_out.name} | "
        f"lignes: {initial_count} -> {after_global_noise} -> "
        f"{after_front} -> {after_end} -> {after_filter} -> {after_dedupe}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    log("START PROGRAMME 2 V2")

    if not TXT_DIR_IN.exists():
        log(f"Répertoire introuvable : {TXT_DIR_IN}")
        return

    txt_files = sorted(TXT_DIR_IN.glob("*.txt"))

    if not txt_files:
        log("Aucun fichier TXT trouvé.")
        return

    TXT_DIR_OUT.mkdir(parents=True, exist_ok=True)

    log(f"{len(txt_files)} fichier(s) trouvé(s).")

    ok = 0
    ko = 0

    for path_in in txt_files:
        try:
            path_out = TXT_DIR_OUT / path_in.name
            process_txt_file(path_in, path_out)
            ok += 1
        except Exception as e:
            ko += 1
            log(f"ERREUR sur {path_in.name} : {type(e).__name__} : {e}")

    log(f"DONE - succès: {ok} | erreurs: {ko}")


if __name__ == "__main__":
    main()