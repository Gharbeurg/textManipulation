from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests


# ============================================================
# PARAMETRES GENERAUX
# ============================================================

TXT_DIR = Path(r"C:/PYTHON/.entree/Sources")
OUTPUT_FILE = Path(r"C:/PYTHON/.data/Synthèse.txt")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen3:8b"

REQUEST_TIMEOUT = 240
OLLAMA_RETRIES = 2

SECTION_MAX_CHARS = 9000
SECTION_OVERLAP = 800

MAX_EXTRACTED_IDEAS_PER_CHUNK = 12
MAX_STRUCTURING_IDEAS = 10
MAX_COMPLEMENTARY_IDEAS = 10
MAX_MISSING_IDEAS = 5


# ============================================================
# LOGS
# ============================================================

def log(message: str) -> None:
    print(
        "{} - {}".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            message
        ),
        flush=True
    )


# ============================================================
# FICHIERS
# ============================================================

def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def init_output_file() -> None:
    ensure_parent_dir(OUTPUT_FILE)
    OUTPUT_FILE.write_text("", encoding="utf-8")


def append_output_block(text: str, add_separator: bool = False) -> None:
    ensure_parent_dir(OUTPUT_FILE)

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        if add_separator:
            f.write("\n" + "=" * 100 + "\n\n")
        f.write(text.strip() + "\n")


# ============================================================
# OUTILS TEXTE
# ============================================================

def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        cleaned = re.sub(r"\s+", " ", item.strip())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def keep_new_items(base_items: List[str], candidate_items: List[str]) -> List[str]:
    seen = {re.sub(r"\s+", " ", x.strip()).lower() for x in base_items if x.strip()}
    result = []

    for item in candidate_items:
        cleaned = re.sub(r"\s+", " ", item.strip())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def split_into_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    results = []

    for part in parts:
        part = part.strip(" -\t")
        if part:
            results.append(part)

    return results


# ============================================================
# APPEL OLLAMA
# ============================================================

def call_ollama(prompt: str, model: str = MODEL_NAME, num_predict: int = 2048) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": num_predict,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def safe_call_ollama(prompt: str, model: str = MODEL_NAME, num_predict: int = 2048) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(1, OLLAMA_RETRIES + 2):
        try:
            log(f"Appel Ollama - tentative {attempt}")
            text = call_ollama(prompt, model=model, num_predict=num_predict)

            if text.strip():
                log("Réponse Ollama reçue")
                return text.strip()

            last_error = RuntimeError("Réponse vide de Ollama.")

        except Exception as exc:
            last_error = exc

        time.sleep(1.2)

    log(f"Ollama indisponible ou réponse vide : {last_error}")
    return ""


# ============================================================
# PARSING DES LISTES
# ============================================================

def parse_bullet_list(raw_text: str) -> List[str]:
    items: List[str] = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.match(r"^[-*•]\s+", line):
            item = re.sub(r"^[-*•]\s+", "", line).strip()
            if item:
                items.append(item)
            continue

        if re.match(r"^\d+[.)]\s+", line):
            item = re.sub(r"^\d+[.)]\s+", "", line).strip()
            if item:
                items.append(item)
            continue

    return dedupe_preserve_order(items)


# ============================================================
# PROMPTS
# ============================================================

def build_extract_prompt(doc_title: str, section_title: str, text: str) -> str:
    return f"""
You analyze a document section and extract only the most important ideas.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- No invented content.
- Keep only the most meaningful ideas.
- Avoid duplication.
- Give at most {MAX_EXTRACTED_IDEAS_PER_CHUNK} ideas.

Document title: {doc_title}
Section title: {section_title}

Section text:
\"\"\"
{text}
\"\"\"
""".strip()


def build_select_structuring_ideas_prompt(doc_title: str, all_ideas: List[str], max_ideas: int) -> str:
    ideas_text = "\n".join([f"- {idea}" for idea in all_ideas])

    return f"""
You receive a list of ideas extracted from a document.

Task:
select the most structuring ideas needed to understand the document.

By "structuring ideas", we mean ideas that provide:
- the topic or objective of the document
- the main findings
- the major results
- the overall conclusion
- the major limitations or implications if they matter

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- You may lightly rephrase for clarity.
- Keep at most {max_ideas} ideas.
- Avoid duplicates and ideas that are too similar.
- Do not invent anything.

Document title: {doc_title}

List of ideas:
\"\"\"
{ideas_text}
\"\"\"
""".strip()


def build_select_complementary_ideas_prompt(
    doc_title: str,
    all_ideas: List[str],
    structuring_ideas: List[str],
    max_ideas: int
) -> str:
    all_ideas_text = "\n".join([f"- {idea}" for idea in all_ideas])
    structuring_text = "\n".join([f"- {idea}" for idea in structuring_ideas])

    return f"""
You receive:
1) the full list of ideas extracted from a document
2) a list already chosen as the most structuring ideas

Task:
select up to {max_ideas} important complementary ideas that add useful information
without repeating the ideas already selected.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- These ideas must be important but not redundant with the structuring ideas.
- They may clarify a method, an important nuance, a limitation, a recommendation, or a useful implication.
- Do not invent anything.

Document title: {doc_title}

Structuring ideas already selected:
\"\"\"
{structuring_text}
\"\"\"

Full list of ideas:
\"\"\"
{all_ideas_text}
\"\"\"
""".strip()


def build_check_missing_ideas_prompt(
    doc_title: str,
    all_ideas: List[str],
    selected_ideas: List[str],
    max_missing: int
) -> str:
    all_ideas_text = "\n".join([f"- {idea}" for idea in all_ideas])
    selected_text = "\n".join([f"- {idea}" for idea in selected_ideas])

    return f"""
You receive:
1) the full list of ideas extracted from a document
2) the current selection of key ideas

Task:
check whether important ideas are still missing from the current selection.

Rules:
- Output in English only.
- Look only for truly important ideas missing from the current selection.
- Do not add redundant ideas.
- Do not add highly secondary ideas.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- Give at most {max_missing} missing ideas.
- If no important idea is missing, respond exactly:
- No important missing idea.
- Do not invent anything.

Document title: {doc_title}

Current selection:
\"\"\"
{selected_text}
\"\"\"

Full list of ideas:
\"\"\"
{all_ideas_text}
\"\"\"
""".strip()


def build_force_english_prompt(raw_text: str) -> str:
    return f"""
Rewrite the following content in English only.

Rules:
- Output only in English.
- Keep the bullet list structure.
- One idea per bullet line starting with "- ".
- Do not add commentary.
- Do not add titles.
- Do not add explanations.
- Do not invent anything.
- If a bullet is already in English, keep it in English and improve only if needed.
- If a bullet is in another language, translate it into natural English.
- If the text contains mixed languages, normalize everything into English only.

Text:
\"\"\"
{raw_text}
\"\"\"
""".strip()


# ============================================================
# NORMALISATION EN ANGLAIS
# ============================================================

def normalize_bullets_to_english(items: List[str]) -> List[str]:
    cleaned_items = [item.strip() for item in items if item.strip()]
    if not cleaned_items:
        return []

    raw_text = "\n".join([f"- {item}" for item in cleaned_items])
    prompt = build_force_english_prompt(raw_text)
    raw = safe_call_ollama(prompt, num_predict=1024)

    if not raw:
        return dedupe_preserve_order(cleaned_items)

    english_items = parse_bullet_list(raw)
    if not english_items:
        return dedupe_preserve_order(cleaned_items)

    return dedupe_preserve_order(english_items)


# ============================================================
# NETTOYAGE DU TEXTE
# ============================================================

EDITORIAL_START_PATTERNS = [
    r"^provided in cooperation with:.*$",
    r"^suggested citation:.*$",
    r"^this version is available at:.*$",
    r"^standard-nutzungsbedingungen:.*$",
    r"^terms of use:.*$",
    r"^received:\s.*$",
    r"^revised:\s.*$",
    r"^accepted:\s.*$",
    r"^published:\s.*$",
    r"^citation:\s.*$",
    r"^copyright:\s.*$",
    r"^licensee .*",
    r"^this article is an open access article.*$",
]

NOISE_LINE_PATTERNS = [
    r"^\s*available online:\s*https?://.*$",
    r"^\s*available at:\s*https?://.*$",
    r"^\s*https?://\S+\s*$",
    r"^\s*\[crossref\]\s*$",
    r"^\s*\[pubmed\]\s*$",
    r"^\s*\[crossref\]\s*\[pubmed\]\s*$",
    r"^\s*doi:\s*.*$",
    r"^\s*orcid:\s*.*$",
    r"^\s*e-?mail:\s*.*$",
    r"^\s*correspondence:\s*.*$",
    r"^\s*submitted\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*$",
    r"^\s*accepted\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*$",
]

END_SECTION_PATTERNS = [
    r"^\s*references\s*:?\s*$",
    r"^\s*bibliography\s*:?\s*$",
    r"^\s*endnotes\s*:?\s*$",
    r"^\s*notes\s*:?\s*$",
    r"^\s*author contributions\s*:?\s*$",
    r"^\s*funding\s*:?\s*$",
    r"^\s*institutional review board statement\s*:?\s*$",
    r"^\s*informed consent statement\s*:?\s*$",
    r"^\s*data availability statement\s*:?\s*$",
    r"^\s*conflicts? of interest\s*:?\s*$",
    r"^\s*acknowledg?ments?\s*:?\s*$",
    r"^\s*appendix\s*:?\s*$",
]

EARLY_STOP_TITLES = {
    "references",
    "bibliography",
    "endnotes",
    "author contributions",
    "funding",
    "institutional review board statement",
    "informed consent statement",
    "data availability statement",
    "conflicts of interest",
    "conflict of interest",
}

def remove_editorial_header_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        lower = line.strip().lower()
        skip = False

        for pattern in EDITORIAL_START_PATTERNS:
            if re.match(pattern, lower, flags=re.IGNORECASE):
                skip = True
                break

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned)


def remove_noise_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        skip = False

        for pattern in NOISE_LINE_PATTERNS:
            if re.match(pattern, line.strip(), flags=re.IGNORECASE):
                skip = True
                break

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned)


def find_end_cut_position(text: str) -> Optional[int]:
    lines = text.splitlines()

    if not lines:
        return None

    total_chars = len(text)
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        for pattern in END_SECTION_PATTERNS:
            if re.match(pattern, stripped, flags=re.IGNORECASE):
                ratio = current_pos / max(total_chars, 1)

                if lower in EARLY_STOP_TITLES:
                    if ratio >= 0.45:
                        return current_pos
                else:
                    if ratio >= 0.65:
                        return current_pos

        current_pos += len(line) + 1

    return None


def merge_broken_lines(text: str) -> str:
    lines = text.splitlines()
    merged: List[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            merged.append(buffer.strip())
        buffer = ""

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_buffer()
            merged.append("")
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        if len(line) <= 2 and re.fullmatch(r"[-–•]", line):
            flush_buffer()
            merged.append(line)
            continue

        looks_like_title = is_section_title(line)

        if looks_like_title:
            flush_buffer()
            merged.append(line)
            continue

        if not buffer:
            buffer = line
            continue

        if buffer.endswith(("-", "–")):
            buffer = buffer[:-1].rstrip() + line
            continue

        if re.search(r"[.:;!?)]$", buffer):
            flush_buffer()
            buffer = line
            continue

        if line[:1].islower():
            buffer += " " + line
            continue

        if len(buffer) < 120:
            buffer += " " + line
            continue

        flush_buffer()
        buffer = line

    flush_buffer()

    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_source_text(raw_text: str) -> str:
    log("Nettoyage du texte brut")

    text = normalize_spaces(raw_text)
    text = remove_editorial_header_lines(text)
    text = remove_noise_lines(text)

    cut_pos = find_end_cut_position(text)
    if cut_pos is not None:
        text = text[:cut_pos].strip()

    text = re.sub(
        r"\nTable\s+\d+.*?(?=\n(?:[A-Z][^\n]{0,120}:|\d+(?:\.\d+)*\.?\s+[A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)*\n|\Z))",
        "\n",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"\nFigure\s+\d+.*?(?=\n(?:[A-Z][^\n]{0,120}:|\d+(?:\.\d+)*\.?\s+[A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)*\n|\Z))",
        "\n",
        text,
        flags=re.DOTALL
    )

    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if re.fullmatch(r"[\d\s,.\-%()/:]+", stripped):
            continue

        if len(stripped) <= 3 and re.fullmatch(r"[\d.%,-]+", stripped):
            continue

        if re.fullmatch(r"page \d+", stripped.lower()):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = merge_broken_lines(text)
    text = normalize_spaces(text)

    return text


# ============================================================
# DECOUPAGE PAR SECTIONS
# ============================================================

SECTION_TITLE_PATTERNS = [
    r"^\d+(\.\d+)*\.\s+.+$",
    r"^\d+(\.\d+)*\s+.+$",
    r"^(abstract|introduction|executive summary|summary|foreword|background|context|method|methods|methodology|results|findings|discussion|conclusion|conclusions|implications|recommendations|limitations|future research directions?|appendix|acknowledg?ments?)\s*$",
]

def is_section_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if len(stripped) > 140:
        return False

    for pattern in SECTION_TITLE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True

    return False


def split_into_sections(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()

    sections: List[Dict[str, str]] = []
    current_title = "Document"
    current_lines: List[str] = []
    found_title = False

    for line in lines:
        if is_section_title(line):
            found_title = True

            if current_lines:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_lines).strip()
                })

            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_lines).strip()
        })

    sections = [s for s in sections if s["text"].strip()]

    if not sections:
        return [{"title": "Document", "text": text.strip()}]

    if not found_title:
        return [{"title": "Document", "text": text.strip()}]

    return sections


def split_large_text(text: str, chunk_size: int = SECTION_MAX_CHARS, overlap: int = SECTION_OVERLAP) -> List[str]:
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)

        if target_end < text_length:
            candidates = [
                text.rfind("\n\n", start, target_end),
                text.rfind(". ", start, target_end),
                text.rfind("; ", start, target_end),
            ]
            candidates = [c for c in candidates if c > start + int(chunk_size * 0.55)]
            end = max(candidates) if candidates else target_end
        else:
            end = target_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_section_chunks(text: str) -> List[Dict[str, str]]:
    sections = split_into_sections(text)
    chunks: List[Dict[str, str]] = []

    for section in sections:
        section_title = section["title"]
        section_text = section["text"].strip()

        if not section_text:
            continue

        sub_chunks = split_large_text(section_text)

        for sub_chunk in sub_chunks:
            if sub_chunk.strip():
                chunks.append({
                    "section_title": section_title,
                    "text": sub_chunk.strip()
                })

    if not chunks and text.strip():
        chunks.append({
            "section_title": "Document",
            "text": text.strip()
        })

    return chunks


# ============================================================
# MODE DE SECOURS SANS OLLAMA
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "as", "at", "from", "it", "its", "their", "there", "than", "into",
    "about", "across", "can", "could", "may", "might", "will", "would", "should",
    "our", "we", "they", "them", "he", "she", "his", "her", "you", "your"
}

def looks_like_reference_fragment(text: str) -> bool:
    low = text.lower()

    if "doi" in low or "orcid" in low or "creativecommons" in low:
        return True

    if re.search(r"\b[a-zà-ÿ-]+ et al\b", low):
        return True

    if low.count(",") > 6 and len(low) < 220:
        return True

    if re.search(r"\bvol\.?\b|\bissue\b|\bjournal\b", low) and re.search(r"\b\d{4}\b", low):
        return True

    return False


def fallback_extract_ideas_from_text(text: str, max_ideas: int = 8) -> List[str]:
    sentences = split_into_sentences(text)
    candidates = []

    for sentence in sentences:
        s = sentence.strip()

        if len(s) < 50:
            continue
        if len(s) > 320:
            continue
        if looks_like_reference_fragment(s):
            continue

        score = 0

        if re.search(r"\b(ai|artificial intelligence|generative ai|genai)\b", s, flags=re.IGNORECASE):
            score += 3
        if re.search(r"\b(result|finding|conclusion|implication|recommendation|evidence|study|review|analysis)\b", s, flags=re.IGNORECASE):
            score += 3
        if re.search(r"\b(productivity|adoption|worker|employee|organization|management|implementation|trust|risk)\b", s, flags=re.IGNORECASE):
            score += 2
        if re.search(r"\b(may|can|could|suggests?|shows?|highlights?|indicates?)\b", s, flags=re.IGNORECASE):
            score += 1

        word_count = len(re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", s))
        score += min(word_count // 12, 3)

        candidates.append((score, s))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [c[1] for c in candidates[:max_ideas]]

    cleaned = []
    for item in selected:
        item = re.sub(r"\s+", " ", item).strip()
        item = item.rstrip(" ;")
        if item:
            cleaned.append(item)

    return dedupe_preserve_order(cleaned)


# ============================================================
# EXTRACTION DES IDEES
# ============================================================

def extract_from_chunks(doc_title: str, chunks: List[Dict[str, str]]) -> List[str]:
    extracted_ideas: List[str] = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        log(f"Extraction des idées - {doc_title} - morceau {index}/{total} - {chunk['section_title']}")

        prompt = build_extract_prompt(doc_title, chunk["section_title"], chunk["text"])
        raw = safe_call_ollama(prompt, num_predict=1536)
        ideas = parse_bullet_list(raw) if raw else []

        if ideas:
            ideas = normalize_bullets_to_english(ideas)
            extracted_ideas.extend(ideas)
            continue

        fallback = fallback_extract_ideas_from_text(chunk["text"], max_ideas=6)

        if fallback:
            extracted_ideas.extend(fallback)
        else:
            extracted_ideas.append("[No idea retrieved for this chunk]")

    return dedupe_preserve_order(extracted_ideas)


def select_structuring_ideas(doc_title: str, all_ideas: List[str], max_ideas: int = MAX_STRUCTURING_IDEAS) -> List[str]:
    log(f"Sélection des {max_ideas} idées les plus structurantes - {doc_title}")

    useful_ideas = [
        idea for idea in all_ideas
        if idea.strip() and idea.strip() != "[No idea retrieved for this chunk]"
    ]

    if not useful_ideas:
        return []

    prompt = build_select_structuring_ideas_prompt(doc_title, useful_ideas, max_ideas)
    raw = safe_call_ollama(prompt, num_predict=1024)
    selected = parse_bullet_list(raw) if raw else []

    if selected:
        selected = normalize_bullets_to_english(selected)
        selected = dedupe_preserve_order(selected)
        if selected:
            return selected[:max_ideas]

    return useful_ideas[:max_ideas]


def select_complementary_ideas(
    doc_title: str,
    all_ideas: List[str],
    structuring_ideas: List[str],
    max_ideas: int = MAX_COMPLEMENTARY_IDEAS
) -> List[str]:
    log(f"Sélection des {max_ideas} idées complémentaires importantes - {doc_title}")

    useful_ideas = [
        idea for idea in all_ideas
        if idea.strip() and idea.strip() != "[No idea retrieved for this chunk]"
    ]

    if not useful_ideas or not structuring_ideas:
        return []

    prompt = build_select_complementary_ideas_prompt(
        doc_title,
        useful_ideas,
        structuring_ideas,
        max_ideas
    )
    raw = safe_call_ollama(prompt, num_predict=1024)
    selected = parse_bullet_list(raw) if raw else []

    if selected:
        selected = normalize_bullets_to_english(selected)
        selected = dedupe_preserve_order(selected)
        selected = keep_new_items(structuring_ideas, selected)
        if selected:
            return selected[:max_ideas]

    remaining = keep_new_items(structuring_ideas, useful_ideas)
    return remaining[:max_ideas]


def check_missing_key_ideas(
    doc_title: str,
    all_ideas: List[str],
    current_selected_ideas: List[str],
    max_missing: int = MAX_MISSING_IDEAS
) -> List[str]:
    log(f"Contrôle des idées importantes manquantes - {doc_title}")

    useful_all_ideas = [
        idea for idea in all_ideas
        if idea.strip() and idea.strip() != "[No idea retrieved for this chunk]"
    ]

    if not useful_all_ideas:
        return []

    if not current_selected_ideas:
        return useful_all_ideas[:max_missing]

    prompt = build_check_missing_ideas_prompt(
        doc_title,
        useful_all_ideas,
        current_selected_ideas,
        max_missing
    )
    raw = safe_call_ollama(prompt, num_predict=1024)
    missing = parse_bullet_list(raw) if raw else []

    if not missing:
        return []

    missing = normalize_bullets_to_english(missing)
    missing = dedupe_preserve_order(missing)
    missing = keep_new_items(current_selected_ideas, missing)

    return missing[:max_missing]


def build_final_key_ideas(
    structuring_ideas: List[str],
    complementary_ideas: List[str],
    missing_ideas: List[str]
) -> List[str]:
    final_ideas: List[str] = []
    final_ideas.extend(structuring_ideas)
    final_ideas.extend(keep_new_items(final_ideas, complementary_ideas))
    final_ideas.extend(keep_new_items(final_ideas, missing_ideas))
    return dedupe_preserve_order(final_ideas)


# ============================================================
# EXTRACTION DES PHRASES AVEC DONNEES CHIFFREES
# ============================================================

NUMBER_PATTERN = re.compile(
    r"""
    (?:
        \b\d+(?:[.,]\d+)?\s*(?:%|‰)
        |
        \b\d{1,3}(?:[ .]\d{3})+(?:[.,]\d+)?
        |
        \b\d+(?:[.,]\d+)?
    )
    """,
    flags=re.VERBOSE
)

METADATA_PATTERNS = [
    r"\bdoi\b",
    r"\borcid\b",
    r"\bissn\b",
    r"\bsubmitted\b",
    r"\brevision\b",
    r"\baccepted\b",
    r"\bpublished\b",
    r"\bcorrespondence\b",
    r"\babstract\b",
    r"\be-?mail\b",
    r"\bvolume\b",
    r"\bissue\b",
    r"\blicense\b",
    r"\bcopyright\b",
    r"\bcreativecommons\b",
    r"\barticle\b",
    r"\bmdpi\b",
    r"\beconstor\b",
    r"https?://",
]

def contains_number_data(text: str) -> bool:
    return bool(NUMBER_PATTERN.search(text))


def is_metadata_sentence(text: str) -> bool:
    low = text.lower()

    for pattern in METADATA_PATTERNS:
        if re.search(pattern, low):
            return True

    if re.fullmatch(r"[\W\d]+", low):
        return True

    return False


def is_date_only_sentence(text: str) -> bool:
    low = text.lower().strip()

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", low):
        return True

    if re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}", low):
        return True

    return False


def clean_numeric_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -\t")
    return text


def extract_numeric_sentences(text: str) -> List[str]:
    log("Extraction des phrases avec données chiffrées")

    sentences = split_into_sentences(text)
    results: List[str] = []

    for sentence in sentences:
        s = clean_numeric_sentence(sentence)

        if not s:
            continue
        if len(s) < 20:
            continue
        if not contains_number_data(s):
            continue
        if re.fullmatch(r"[\d\s,.\-:%()/%]+", s):
            continue
        if is_metadata_sentence(s):
            continue
        if is_date_only_sentence(s):
            continue
        if looks_like_reference_fragment(s):
            continue

        results.append(s)

    return dedupe_preserve_order(results)


# ============================================================
# FORMATAGE
# ============================================================

def format_output_block(
    title: str,
    all_ideas: List[str],
    final_key_ideas: List[str],
    numeric_sentences: List[str]
) -> str:
    blocks = [title]

    blocks.append("ALL EXTRACTED IDEAS")
    if all_ideas:
        blocks.extend([f"- {idea}" for idea in all_ideas])
    else:
        blocks.append("- No usable idea extracted.")

    blocks.append("")
    blocks.append("FINAL SYNTHESIS OF KEY IDEAS")
    if final_key_ideas:
        blocks.extend([f"- {idea}" for idea in final_key_ideas])
    else:
        blocks.append("- No final key idea available.")

    blocks.append("")
    blocks.append("SENTENCES WITH NUMERICAL DATA")
    if numeric_sentences:
        blocks.extend([f"- {sentence}" for sentence in numeric_sentences])
    else:
        blocks.append("- No sentence with numerical data found.")

    return "\n".join(blocks) + "\n"


# ============================================================
# TRAITEMENT D'UN FICHIER
# ============================================================

def process_file(file_path: Path) -> str:
    log(f"Lecture du fichier : {file_path}")
    raw_content = read_text_file(file_path)

    if not raw_content.strip():
        log(f"Fichier vide : {file_path.name}")
        return ""

    title = file_path.stem

    clean_content = clean_source_text(raw_content)
    if not clean_content.strip():
        log(f"Contenu vide après nettoyage : {file_path.name}")
        return f"{title}\nALL EXTRACTED IDEAS\n- Content empty after cleaning.\n"

    log(f"Taille texte nettoyé : {len(clean_content)} caractères")

    chunks = build_section_chunks(clean_content)
    log(f"Nombre de morceaux pour {file_path.name} : {len(chunks)}")

    all_ideas = extract_from_chunks(title, chunks)
    all_ideas = dedupe_preserve_order(all_ideas)

    structuring_ideas = select_structuring_ideas(
        title,
        all_ideas,
        max_ideas=MAX_STRUCTURING_IDEAS
    )

    complementary_ideas = select_complementary_ideas(
        title,
        all_ideas,
        structuring_ideas,
        max_ideas=MAX_COMPLEMENTARY_IDEAS
    )

    current_selected = []
    current_selected.extend(structuring_ideas)
    current_selected.extend(keep_new_items(current_selected, complementary_ideas))

    missing_ideas = check_missing_key_ideas(
        title,
        all_ideas,
        current_selected,
        max_missing=MAX_MISSING_IDEAS
    )

    final_key_ideas = build_final_key_ideas(
        structuring_ideas,
        complementary_ideas,
        missing_ideas
    )
    final_key_ideas = normalize_bullets_to_english(final_key_ideas)

    numeric_sentences = extract_numeric_sentences(clean_content)

    return format_output_block(
        title=title,
        all_ideas=all_ideas,
        final_key_ideas=final_key_ideas,
        numeric_sentences=numeric_sentences
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("Démarrage du script")

    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {TXT_DIR}")

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    log(f"Nombre de fichiers trouvés : {len(txt_files)}")

    if not txt_files:
        log("Aucun fichier .txt trouvé")
        return

    init_output_file()
    log(f"Fichier de sortie initialisé : {OUTPUT_FILE}")

    first_written = False

    for file_path in txt_files:
        try:
            log(f"Début traitement fichier : {file_path.name}")
            block = process_file(file_path)

            if block.strip():
                append_output_block(block, add_separator=first_written)
                first_written = True
                log(f"Bloc écrit dans le fichier de synthèse : {file_path.name}")

            log(f"Fin traitement fichier : {file_path.name}")

        except Exception as exc:
            print(
                "{} - ERREUR {} : {}".format(
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    file_path.name,
                    exc
                ),
                file=sys.stderr,
                flush=True
            )

    log(f"Fichier de sortie mis à jour au fil de l'eau : {OUTPUT_FILE}")
    log("Fin du script")


if __name__ == "__main__":
    main()