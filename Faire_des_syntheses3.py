# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import re
import time

import requests


# ============================================================
# PARAMETRES
# ============================================================

TXT_DIR = Path(r"C:/PYTHON/.entree/SourcesSYNTHESE")
OUTPUT_DIR = Path(r"C:/PYTHON/.data/ResultatsIdees")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:8b"

OUTPUT_LANGUAGE = "English"

TARGET_CHUNK_WORDS = 1200
MAX_CHUNK_WORDS = 1600
MIN_CHUNK_WORDS = 350

REQUEST_TIMEOUT = 240
MAX_NUMERIC_SENTENCES = 400
MIN_NUMERIC_SCORE = 1


# ============================================================
# LOG
# ============================================================

def log(label: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {label}", flush=True)


# ============================================================
# OUTILS TEXTE
# ============================================================

SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=",
})

def clear_output_directory(directory: Path) -> None:
    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()  # supprime le fichier

def normalize_scientific_notation(text: str) -> str:
    text = text.translate(SUPERSCRIPT_MAP)

    # 10−7, 10-7, 10+7 => 10^-7 / 10^+7 quand il s'agit clairement d'un exposant collé
    text = re.sub(r"(?<![A-Za-z0-9])10\s*([-+])\s*(\d{1,3})(?![\d/])", r"10^\1\2", text)

    # 1 × 10-7 => 1 × 10^-7
    text = re.sub(r"([×x])\s*10\s*([-+])\s*(\d{1,3})(?![\d/])", r"\1 10^\2\3", text)

    # 10 7 seul après x/× est parfois un exposant mal normalisé ; on n'y touche pas sans signe pour éviter les faux positifs
    return text


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\u2212", "-")   # unicode minus
    text = text.replace("\u2013", "-")   # en dash
    text = text.replace("\u2014", "-")   # em dash
    text = normalize_scientific_notation(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def strip_bullets_prefix(text: str) -> str:
    return re.sub(r"^\s*[-•*\d\.\)\(]+\s*", "", text).strip()


def clean_idea_text(text: str) -> str:
    text = text.strip()
    text = strip_bullets_prefix(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -•*")
    return text


def split_into_sentences_basic(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def deduplicate_texts(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        cleaned = re.sub(r"\s+", " ", item.strip())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def deduplicate_ideas_simple(ideas: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for idea in ideas:
        normalized = re.sub(r"[^a-z0-9]+", " ", idea.lower()).strip()
        if not normalized:
            continue

        tokens = normalized.split()
        key = " ".join(tokens[:20])

        if key in seen:
            continue

        seen.add(key)
        out.append(idea)

    return out


# ============================================================
# APPEL OLLAMA
# ============================================================

def ollama_generate(prompt: str, temperature: float = 0.2, num_predict: int = 1200) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


# ============================================================
# PARSING TEXTE STRUCTURE
# ============================================================

def parse_structured_text(text: str) -> Dict:
    lines = [line.rstrip() for line in text.splitlines()]

    title = ""
    abstract_parts: List[str] = []
    keywords = ""
    sections: List[Dict] = []

    current_section = None
    current_subsection = None
    current_mode = "body"

    def ensure_section_if_needed():
        nonlocal current_section, sections
        if current_section is None:
            current_section = {
                "title": "Untitled section",
                "subsections": [],
                "content": []
            }
            sections.append(current_section)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("TITLE: "):
            if not title:
                title = line.replace("TITLE: ", "", 1).strip()
            continue

        if line == "ABSTRACT":
            current_mode = "abstract"
            continue

        if line == "KEYWORDS":
            current_mode = "keywords"
            continue

        if line.startswith("SECTION: "):
            current_mode = "body"
            current_subsection = None
            current_section = {
                "title": line.replace("SECTION: ", "", 1).strip(),
                "subsections": [],
                "content": []
            }
            sections.append(current_section)
            continue

        if line.startswith("SUBSECTION: "):
            current_mode = "body"
            ensure_section_if_needed()
            current_subsection = {
                "title": line.replace("SUBSECTION: ", "", 1).strip(),
                "content": []
            }
            current_section["subsections"].append(current_subsection)
            continue

        if line.startswith("TABLE: "):
            table_line = line.replace("TABLE: ", "", 1).strip()
            ensure_section_if_needed()
            if current_subsection is not None:
                current_subsection["content"].append(f"[TABLE] {table_line}")
            else:
                current_section["content"].append(f"[TABLE] {table_line}")
            continue

        if line.startswith("FIGURE: "):
            figure_line = line.replace("FIGURE: ", "", 1).strip()
            ensure_section_if_needed()
            if current_subsection is not None:
                current_subsection["content"].append(f"[FIGURE] {figure_line}")
            else:
                current_section["content"].append(f"[FIGURE] {figure_line}")
            continue

        if current_mode == "abstract":
            abstract_parts.append(line)
        elif current_mode == "keywords":
            keywords = (keywords + " " + line).strip()
        else:
            ensure_section_if_needed()
            if current_subsection is not None:
                current_subsection["content"].append(line)
            else:
                current_section["content"].append(line)

    return {
        "title": title.strip(),
        "abstract": normalize_text("\n".join(abstract_parts)),
        "keywords": normalize_text(keywords),
        "sections": sections
    }


def build_document_plan(doc: Dict) -> str:
    lines: List[str] = []

    if doc["title"]:
        lines.append(f"TITLE: {doc['title']}")

    if doc["abstract"]:
        lines.append("ABSTRACT: present")

    if doc["keywords"]:
        lines.append("KEYWORDS: present")

    for i, section in enumerate(doc["sections"], start=1):
        lines.append(f"{i}. {section['title']}")
        for j, subsection in enumerate(section["subsections"], start=1):
            lines.append(f"   {i}.{j} {subsection['title']}")

    return "\n".join(lines).strip()


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(text: str, target_words: int, max_words: int, min_words: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        wc = word_count(para)

        if current_words + wc <= max_words:
            current.append(para)
            current_words += wc
        else:
            if current:
                chunks.append("\n".join(current).strip())
            current = [para]
            current_words = wc

    if current:
        chunks.append("\n".join(current).strip())

    merged: List[str] = []
    buffer = ""

    for ch in chunks:
        if not buffer:
            buffer = ch
            continue

        if word_count(buffer) < min_words:
            buffer = buffer + "\n" + ch
        else:
            merged.append(buffer.strip())
            buffer = ch

    if buffer:
        merged.append(buffer.strip())

    return merged


def build_section_chunks(doc: Dict) -> List[Dict]:
    chunks: List[Dict] = []

    if doc["abstract"]:
        chunks.append({
            "section_title": "Abstract",
            "subsection_title": "",
            "text": doc["abstract"]
        })

    for section in doc["sections"]:
        section_title = section["title"]

        if section["content"]:
            section_text = normalize_text("\n".join(section["content"]))
            for ch in split_text_into_chunks(section_text, TARGET_CHUNK_WORDS, MAX_CHUNK_WORDS, MIN_CHUNK_WORDS):
                chunks.append({
                    "section_title": section_title,
                    "subsection_title": "",
                    "text": ch
                })

        for subsection in section["subsections"]:
            subsection_text = normalize_text("\n".join(subsection["content"]))
            if not subsection_text:
                continue

            for ch in split_text_into_chunks(subsection_text, TARGET_CHUNK_WORDS, MAX_CHUNK_WORDS, MIN_CHUNK_WORDS):
                chunks.append({
                    "section_title": section_title,
                    "subsection_title": subsection["title"],
                    "text": ch
                })

    return chunks


# ============================================================
# PROMPTS OLLAMA
# ============================================================

def build_chunk_prompt(doc_title: str, plan: str, chunk: Dict) -> str:
    section_label = chunk["section_title"]
    if chunk["subsection_title"]:
        section_label += f" > {chunk['subsection_title']}"

    return f"""
You are extracting all important ideas from a document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

CURRENT SECTION:
{section_label}

TASK:
Extract all important ideas contained in the text below.

RULES:
- Extract concrete ideas only.
- Do not write vague sentences such as "the article discusses" or "the text presents".
- Keep factual findings, arguments, mechanisms, distinctions, implications, and important methodological points.
- Keep useful numbers when they matter for the idea.
- Ignore metadata, repeated headers, page numbers, references, decorative labels, and layout noise.
- Each idea must be a full sentence.
- One idea per line, starting with "- ".
- Do not merge unrelated ideas into one sentence.

TEXT:
{chunk["text"]}
""".strip()


def build_missing_ideas_prompt(doc_title: str, plan: str, chunk: Dict, existing_ideas: List[str]) -> str:
    section_label = chunk["section_title"]
    if chunk["subsection_title"]:
        section_label += f" > {chunk['subsection_title']}"

    joined_ideas = "\n".join(f"- {idea}" for idea in existing_ideas) if existing_ideas else "- None"

    return f"""
You are checking whether important ideas were missed in a document extraction step.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

CURRENT SECTION:
{section_label}

ALREADY EXTRACTED IDEAS:
{joined_ideas}

TASK:
Read the text below and return only the important ideas that are missing from the existing list.

RULES:
- Return only missing ideas.
- Do not repeat any idea already present.
- Keep only concrete and useful ideas.
- Ignore metadata, references, headers, page numbers, and layout noise.
- Each idea must be a full sentence.
- One idea per line, starting with "- ".
- If nothing important is missing, return exactly:
NONE

TEXT:
{chunk["text"]}
""".strip()


def build_consolidation_prompt(doc_title: str, plan: str, ideas: List[str]) -> str:
    joined = "\n".join(f"- {x}" for x in ideas)

    return f"""
You are consolidating a list of ideas extracted from a document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

TASK:
Clean and deduplicate the list below while preserving all distinct important ideas.

RULES:
- Keep all important distinct ideas.
- Remove duplicates and near-duplicates.
- Do not remove an idea just because it is shorter.
- Preserve useful numbers and contrasts.
- Keep the wording concrete.
- Each idea must be a full sentence.
- One idea per line, starting with "- ".

IDEAS:
{joined}
""".strip()


# ============================================================
# PARSING SORTIE OLLAMA
# ============================================================

def parse_bullet_lines(text: str) -> List[str]:
    lines: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.upper() == "NONE":
            continue

        if line.startswith("- "):
            idea = clean_idea_text(line[2:])
            if idea:
                lines.append(idea)
        else:
            idea = clean_idea_text(line)
            if idea:
                lines.append(idea)

    return lines


# ============================================================
# EXTRACTION PHRASES CHIFFREES
# ============================================================

NUMBER_PATTERN = re.compile(
    r"""
    (?:
        # scientific notation: 1e-3, 2E+6
        \b\d+(?:[.,]\d+)?[eE][+-]?\d+\b
        |
        # powers: 10^5, 10^-12
        \b\d+(?:[.,]\d+)?\s*\^\s*[+-]?\d+\b
        |
        # times ten: 1 × 10^7, 6 x 10^8
        \b\d+(?:[.,]\d+)?\s*[×x]\s*10\s*\^\s*[+-]?\d+\b
        |
        # unicode exponent already normalised when possible
        \b\d+(?:[.,]\d+)?\s*[×x]\s*10\s*[-+]\s*\d+\b
        |
        # uncertainty notation: 65.6(1.3)
        \b\d+(?:[.,]\d+)?\(\d+(?:[.,]\d+)?\)\b
        |
        # ± notation
        \b\d+(?:[.,]\d+)?\s*(?:±|\+/-)\s*\d+(?:[.,]\d+)?\b
        |
        # percentages
        \b\d+(?:[.,]\d+)?\s*(?:%|‰)\b
        |
        # ranges: 3-5, 3.2-4.1
        \b\d+(?:[.,]\d+)?\s*[-]\s*\d+(?:[.,]\d+)?\b
        |
        # grouped thousands
        \b\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?\b
        |
        # plain numbers
        \b\d+(?:[.,]\d+)?\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
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
    r"\be-?mail\b",
    r"\bvolume\b",
    r"\bissue\b",
    r"\blicense\b",
    r"\bcopyright\b",
    r"\bcopyright holder\b",
    r"\bcopyright owner\b",
    r"\bcopyright notice\b",
    r"\bcopyright year\b",
    r"\bcreativecommons\b",
    r"https?://",
]

MONTHS_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE
)

ABBREVIATIONS_PATTERN = re.compile(
    r"""
    \b(
        fig|figs|eq|eqs|ref|refs|dr|mr|mrs|ms|prof|inc|ltd|jr|sr|vs|no|nos|al
    )\.
    """,
    flags=re.IGNORECASE | re.VERBOSE
)


REFERENCE_ONLY_PREFIX = re.compile(
    r"""
    ^\s*(?:
        \[\d+(?:\s*,\s*\d+)*\]
        |
        \d+\.
        |
        refs?\.?
    )\s+
    """,
    flags=re.IGNORECASE | re.VERBOSE
)

def contains_number_data(text: str) -> bool:
    return bool(NUMBER_PATTERN.search(text))


def is_metadata_sentence(sentence: str) -> bool:
    low = sentence.lower()
    return any(re.search(pattern, low) for pattern in METADATA_PATTERNS)


def is_date_only_sentence(sentence: str) -> bool:
    s = sentence.strip()

    if not contains_number_data(s):
        return False

    tmp = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", s)
    tmp = re.sub(r"\b(?:19|20)\d{2}\b", " ", tmp)
    tmp = MONTHS_PATTERN.sub(" ", tmp)

    if NUMBER_PATTERN.search(tmp):
        return False

    return True


def looks_like_reference_fragment(sentence: str) -> bool:
    low = sentence.lower()

    if re.search(r"\bet al\b", low) and "(" in sentence and ")" in sentence:
        return True

    if re.search(r"\b[a-zà-ÿ-]+,\s*[A-Z]\.", sentence):
        return True

    if low.count(";") >= 3 and low.count("(") >= 2:
        return True

    return False


def looks_like_layout_fragment(text: str) -> bool:
    s = text.strip()

    if not s:
        return True

    if re.fullmatch(r"[\d\s,.;:()/%\-+±=×x^]+", s):
        return True

    if re.fullmatch(r"\d{1,4}", s):
        return True

    if re.search(r"^\s*page\s+\d+\s*$", s, flags=re.IGNORECASE):
        return True

    return False


def protect_abbreviations(text: str) -> str:
    text = re.sub(r"\bet al\.", "et al", text, flags=re.IGNORECASE)
    text = re.sub(r"\be\.g\.", "eg", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.", "ie", text, flags=re.IGNORECASE)
    text = ABBREVIATIONS_PATTERN.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    return text


def restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def split_into_sentences_numeric(text: str) -> List[str]:
    text = normalize_text(text)
    text = protect_abbreviations(text)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: List[str] = []

    for para in paragraphs:
        parts = re.split(
            r"(?<=[\.\!\?])\s+(?=[A-ZÀ-ÖØ-Ý0-9(\[])|(?<=;)\s+(?=[A-ZÀ-ÖØ-Ý0-9(\[])",
            para
        )

        for part in parts:
            part = restore_abbreviations(part).strip()
            if part:
                sentences.append(part)

    return sentences


def clean_numeric_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)
    sentence = re.sub(r"\s+\)", ")", sentence)
    sentence = re.sub(r"\(\s+", "(", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.strip(" -•\t;,")
    sentence = REFERENCE_ONLY_PREFIX.sub("", sentence).strip()
    return sentence


def sentence_seems_incomplete(sentence: str) -> bool:
    s = sentence.strip()

    if not s:
        return True

    if re.search(r"[:;,(\[]\s*$", s):
        return True

    if len(s) < 30:
        return True

    if s.lower().startswith(("and ", "or ", "but ", "whereas ", "while ", "with ", "including ", "such as ")):
        return True

    if re.search(r"\b(compared with|respectively|where|which|that|while|whereas|including)\s*$", s.lower()):
        return True

    return False


def sentence_starts_like_continuation(sentence: str) -> bool:
    s = sentence.strip()

    if not s:
        return False

    if s[0].islower():
        return True

    if s.startswith((")", "]", ",", ";", ":", "%")):
        return True

    if s.lower().startswith(("and ", "or ", "but ", "whereas ", "while ", "with ", "including ", "respectively")):
        return True

    return False


def merge_broken_numeric_sentences(sentences: List[str]) -> List[str]:
    if not sentences:
        return []

    merged: List[str] = []
    i = 0

    while i < len(sentences):
        current = sentences[i].strip()

        while i + 1 < len(sentences):
            nxt = sentences[i + 1].strip()

            should_merge = False

            if contains_number_data(current) and sentence_seems_incomplete(current):
                should_merge = True
            elif contains_number_data(nxt) and sentence_starts_like_continuation(nxt):
                should_merge = True
            elif contains_number_data(current) and contains_number_data(nxt) and len(current) < 45:
                should_merge = True

            if not should_merge:
                break

            current = f"{current} {nxt}".strip()
            i += 1

        merged.append(clean_numeric_sentence(current))
        i += 1

    return merged


def extract_numeric_candidates_from_paragraph(paragraph: str) -> List[str]:
    raw_sentences = split_into_sentences_numeric(paragraph)
    merged_sentences = merge_broken_numeric_sentences(raw_sentences)

    kept: List[str] = []

    for sentence in merged_sentences:
        s = clean_numeric_sentence(sentence)

        if not s:
            continue
        if not contains_number_data(s):
            continue
        if looks_like_layout_fragment(s):
            continue
        if is_metadata_sentence(s):
            continue
        if is_date_only_sentence(s):
            continue
        if looks_like_reference_fragment(s):
            continue

        kept.append(s)

    return kept


def canonical_numeric_key(text: str) -> str:
    s = text.lower()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def count_alpha_words(text: str) -> int:
    return len(re.findall(r"\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*\b", text))


def count_digits(text: str) -> int:
    return len(re.findall(r"\d", text))


def has_scientific_unit(text: str) -> bool:
    return bool(re.search(
        r"""
        (
            %|‰|
            \b(?:k|m|cm|mm|um|µm|nm|pm)\b|
            \b(?:g|kg|mg|ug|µg)\b|
            \b(?:s|ms|us|µs|ns|ps|h|hr)\b|
            \b(?:hz|khz|mhz|ghz)\b|
            \b(?:ev|kev|mev|gev|tev)\b|
            \b(?:v|mv|kv|a|ma|ua|µa|w|mw|kw)\b|
            \b(?:t|mt|kt)\b|
            \b(?:pa|kpa|mpa|bar|mbar)\b|
            \b(?:l|ml|ul|µl)\b|
            \b(?:mol|mmol)\b|
            \b(?:°c)\b|
            \b(?:km/h|km h-1|m/s|m s-1|m s-2|kg/m3|g/cm3|mev/c|mev/c2|kev/c|ppb|ppt|ppm|p\.p\.b\.|p\.p\.t\.|p\.p\.m\.)\b
        )
        """,
        text,
        flags=re.IGNORECASE | re.VERBOSE
    ))


def has_measurement_pattern(text: str) -> bool:
    return bool(re.search(
        r"""
        (
            \b\d+(?:[.,]\d+)?\s*(?:±|\+/-)\s*\d+(?:[.,]\d+)?\b
            |
            \b\d+(?:[.,]\d+)?\(\d+(?:[.,]\d+)?\)\b
            |
            \b\d+(?:[.,]\d+)?\s*[×x]\s*10\s*\^\s*[+-]?\d+\b
            |
            \b10\s*\^\s*[+-]?\d+\b
            |
            \b\d+(?:[.,]\d+)?[eE][+-]?\d+\b
        )
        """,
        text,
        flags=re.IGNORECASE | re.VERBOSE
    ))


def has_action_verb(text: str) -> bool:
    return bool(re.search(
        r"""
        \b(
            is|are|was|were|be|been|being|
            has|have|had|
            shows|showed|shown|
            found|finds|reported|measured|observed|
            increased|decreased|improved|reduced|
            reached|yielded|gave|obtained|transported|
            consumed|lasted|remained|corresponded|
            produced|generated|detected|achieved|
            provide|provides|provided|allow|allows|allowed
        )\b
        """,
        text,
        flags=re.IGNORECASE | re.VERBOSE
    ))


def starts_like_reference_entry(text: str) -> bool:
    s = text.strip()
    return bool(re.match(r"^\d+\.\s+[A-Z][a-zA-Z-]+", s))


def looks_like_journal_reference(text: str) -> bool:
    s = text.strip()

    if re.search(r"\bet al\b", s, flags=re.IGNORECASE) and re.search(r"\(\d{4}\)", s):
        return True

    if re.search(r"\bNature\b|\bScience\b|\bPhys\.\s*Rev\.?\b|\bRev\.\s*Mod\.\s*Phys\.?\b|\bLett\.?\b", s):
        if re.search(r"\(\d{4}\)", s) or re.search(r"\b\d+\s*,\s*\d+\s*-\s*\d+\b", s):
            return True

    if re.search(r"\b\d+\s*,\s*\d+\s*-\s*\d+\s*\(\d{4}\)", s):
        return True

    return False


def looks_like_figure_caption(text: str) -> bool:
    s = text.strip()

    if re.match(r"^(figure|fig\.?|figs\.?|table|extended data fig\.?|extended data table)\s*\d+[a-z]?\b",
                s, flags=re.IGNORECASE):
        return True

    if s.lower().startswith("figure:"):
        return True

    return False


def looks_like_axis_or_table_fragment(text: str) -> bool:
    s = text.strip()

    if s.count("|") >= 2:
        return True

    if re.search(r"\b(temperature|acceleration|time|counts?|frequency|signal|intensity|voltage|current|amplitude|number of protons)\s*\([^)]+\)",
                 s, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[A-Za-z ()/%.\-+]+\s+[A-Za-z ()/%.\-+]+\s+[A-Za-z ()/%.\-+]+", s):
        if count_digits(s) == 0:
            return True

    if re.search(r"\b-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\b", s) and count_alpha_words(s) <= 8:
        return True

    return False


def looks_like_page_counter(text: str) -> bool:
    s = text.strip()

    if re.fullmatch(r"\d+\s+of\s+\d+", s, flags=re.IGNORECASE):
        return True

    if re.search(r"\b\d+\s+of\s+\d+\b", s, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", s, flags=re.IGNORECASE):
        return True

    return False


def looks_like_numbered_section_title(text: str) -> bool:
    s = text.strip()

    if len(s) > 140:
        return False

    if re.match(r"^\d+(\.\d+){1,3}\s+[A-Z]", s):
        return True

    if re.match(r"^(section|subsection|appendix|supplementary|extended data)\b", s, flags=re.IGNORECASE):
        return True

    return False


def is_fragment_too_short(text: str) -> bool:
    s = text.strip()
    words = count_alpha_words(s)

    if words >= 7:
        return False

    if has_scientific_unit(s) or has_measurement_pattern(s):
        return False

    if words <= 4:
        return True

    return False


def numeric_sentence_quality_score(sentence: str) -> int:
    s = sentence.strip()
    low = s.lower()
    score = 0

    # Signaux positifs
    if len(s) >= 60:
        score += 2
    elif len(s) >= 35:
        score += 1

    if count_alpha_words(s) >= 10:
        score += 2
    elif count_alpha_words(s) >= 6:
        score += 1

    if has_scientific_unit(s):
        score += 3

    if has_measurement_pattern(s):
        score += 3

    if has_action_verb(s):
        score += 2

    if "," in s and count_alpha_words(s) >= 8:
        score += 1

    if re.search(r"\b(compared with|respectively|corresponding to|resulted in|consistent with|better than|precision of|factor of)\b", low):
        score += 1

    # Signaux négatifs
    if looks_like_page_counter(s):
        score -= 8

    if looks_like_figure_caption(s):
        score -= 7

    if looks_like_journal_reference(s):
        score -= 7

    if starts_like_reference_entry(s):
        score -= 6

    if looks_like_numbered_section_title(s):
        score -= 5

    if looks_like_axis_or_table_fragment(s):
        score -= 5

    if looks_like_reference_fragment(s):
        score -= 4

    if looks_like_layout_fragment(s):
        score -= 4

    if is_fragment_too_short(s):
        score -= 4

    alpha_chars = len(re.findall(r"[A-Za-zÀ-ÿ]", s))
    digit_chars = len(re.findall(r"\d", s))
    if alpha_chars < 10 and digit_chars >= 3:
        score -= 4

    if s.count(";") >= 2:
        score -= 2

    if s.count(",") >= 5 and count_alpha_words(s) <= 8:
        score -= 3

    return score


def deduplicate_numeric_sentences_keep_order(items: List[Tuple[str, int]]) -> List[str]:
    best_by_key: Dict[str, Tuple[str, int, int]] = {}

    for idx, (sentence, score) in enumerate(items):
        key = canonical_numeric_key(sentence)
        current = best_by_key.get(key)

        if current is None:
            best_by_key[key] = (sentence, score, idx)
            continue

        current_sentence, current_score, current_idx = current
        if score > current_score:
            best_by_key[key] = (sentence, score, current_idx)
        elif score == current_score and len(sentence) > len(current_sentence):
            best_by_key[key] = (sentence, score, current_idx)

    ordered = sorted(best_by_key.values(), key=lambda x: x[2])
    return [x[0] for x in ordered]


def post_clean_numeric_sentences(sentences: List[str]) -> List[str]:
    kept: List[Tuple[str, int]] = []

    for sentence in sentences:
        s = clean_numeric_sentence(sentence)

        if not s:
            continue
        if not contains_number_data(s):
            continue
        if is_metadata_sentence(s):
            continue
        if is_date_only_sentence(s):
            continue

        score = numeric_sentence_quality_score(s)

        if score < MIN_NUMERIC_SCORE:
            continue

        kept.append((s, score))

    return deduplicate_numeric_sentences_keep_order(kept)


def paragraph_has_substantial_numeric_sentence(para: str, collected: List[str]) -> bool:
    para_key = canonical_numeric_key(para)
    for sent in collected:
        if len(sent) < 25:
            continue
        if canonical_numeric_key(sent) in para_key:
            return True
    return False


def extract_numeric_sentences(text: str) -> List[str]:
    log("Extraction robuste des phrases avec données chiffrées")

    text = normalize_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    collected: List[str] = []

    # Passe 1 : extraction large par paragraphe
    for para in paragraphs:
        if not contains_number_data(para):
            continue

        candidates = extract_numeric_candidates_from_paragraph(para)
        collected.extend(candidates)

    # Passe 2 : contrôle de rappel limité aux paragraphes sans vraie phrase retenue
    fallback_collected: List[str] = []

    for para in paragraphs:
        if not contains_number_data(para):
            continue

        if paragraph_has_substantial_numeric_sentence(para, collected):
            continue

        lines = [clean_numeric_sentence(x) for x in para.split("\n") if x.strip()]
        for line in lines:
            if not contains_number_data(line):
                continue
            if looks_like_layout_fragment(line):
                continue
            if is_metadata_sentence(line):
                continue
            if is_date_only_sentence(line):
                continue
            if looks_like_page_counter(line):
                continue
            if looks_like_figure_caption(line):
                continue
            if looks_like_journal_reference(line):
                continue

            fallback_collected.append(line)

    all_sentences = collected + fallback_collected

    # Passe 3 : nettoyage + score en gardant l'ordre du document
    cleaned = post_clean_numeric_sentences(all_sentences)

    return cleaned[:MAX_NUMERIC_SENTENCES]


# ============================================================
# EXTRACTION IDEES
# ============================================================

def extract_ideas_from_chunk(doc_title: str, plan: str, chunk: Dict) -> List[str]:
    prompt = build_chunk_prompt(doc_title, plan, chunk)
    response = ollama_generate(prompt, temperature=0.2, num_predict=1000)

    ideas = parse_bullet_lines(response)
    if not ideas:
        ideas = split_into_sentences_basic(response)

    ideas = [clean_idea_text(x) for x in ideas if clean_idea_text(x)]
    return deduplicate_ideas_simple(ideas)


def extract_missing_ideas_from_chunk(doc_title: str, plan: str, chunk: Dict, existing_ideas: List[str]) -> List[str]:
    prompt = build_missing_ideas_prompt(doc_title, plan, chunk, existing_ideas)
    response = ollama_generate(prompt, temperature=0.1, num_predict=800)

    if response.strip().upper() == "NONE":
        return []

    ideas = parse_bullet_lines(response)
    ideas = [clean_idea_text(x) for x in ideas if clean_idea_text(x)]
    return deduplicate_ideas_simple(ideas)


def consolidate_ideas(doc_title: str, plan: str, ideas: List[str]) -> List[str]:
    if not ideas:
        return []

    prompt = build_consolidation_prompt(doc_title, plan, ideas)
    response = ollama_generate(prompt, temperature=0.1, num_predict=1400)

    consolidated = parse_bullet_lines(response)
    if not consolidated:
        consolidated = ideas[:]

    consolidated = [clean_idea_text(x) for x in consolidated if clean_idea_text(x)]
    return deduplicate_ideas_simple(consolidated)


# ============================================================
# SORTIE
# ============================================================

def build_output_text(doc_title: str, ideas: List[str], numeric_sentences: List[str]) -> str:
    lines: List[str] = []

    lines.append(f"TITRE : {doc_title or 'Unknown title'}")
    lines.append("")

    lines.append("LES IDEES DU DOCUMENT :")
    if ideas:
        for idea in ideas:
            lines.append(f"- {idea}")
    else:
        lines.append("- No idea extracted.")
    lines.append("")

    lines.append("LES CHIFFRES DU DOCUMENT :")
    if numeric_sentences:
        for sentence in numeric_sentences:
            lines.append(f"- {sentence}")
    else:
        lines.append("- No numerical sentence found.")

    return "\n".join(lines).strip() + "\n"


def safe_output_filename(txt_path: Path) -> Path:
    return OUTPUT_DIR / f"{txt_path.stem}_extraction.txt"


# ============================================================
# PIPELINE DOCUMENT
# ============================================================

def process_document(txt_path: Path) -> None:
    log(f"Début traitement : {txt_path.name}")

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)

    doc = parse_structured_text(text)
    plan = build_document_plan(doc)
    chunks = build_section_chunks(doc)

    if not chunks:
        chunks = [{
            "section_title": "Full document",
            "subsection_title": "",
            "text": text
        }]

    log(f"Nombre de chunks : {len(chunks)}")

    all_ideas: List[str] = []

    # Passe 1 : extraction initiale
    for i, chunk in enumerate(chunks, start=1):
        log(f"Extraction initiale idées chunk {i}/{len(chunks)} - {txt_path.name}")
        ideas = extract_ideas_from_chunk(doc["title"], plan, chunk)
        all_ideas.extend(ideas)
        time.sleep(0.2)

    all_ideas = deduplicate_ideas_simple(all_ideas)
    log(f"Nombre d'idées après passe 1 : {len(all_ideas)}")

    # Passe 2 : contrôle des idées manquantes
    missing_ideas_all: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        log(f"Contrôle idées manquantes chunk {i}/{len(chunks)} - {txt_path.name}")
        missing_ideas = extract_missing_ideas_from_chunk(doc["title"], plan, chunk, all_ideas + missing_ideas_all)
        missing_ideas_all.extend(missing_ideas)
        time.sleep(0.2)

    missing_ideas_all = deduplicate_ideas_simple(missing_ideas_all)
    log(f"Idées ajoutées après contrôle : {len(missing_ideas_all)}")

    final_ideas = deduplicate_ideas_simple(all_ideas + missing_ideas_all)
    log(f"Nombre d'idées avant consolidation : {len(final_ideas)}")

    # Consolidation finale
    final_ideas = consolidate_ideas(doc["title"], plan, final_ideas)
    log(f"Nombre d'idées après consolidation : {len(final_ideas)}")

    # Extraction chiffres
    numeric_sentences = extract_numeric_sentences(text)
    log(f"Nombre de phrases chiffrées extraites : {len(numeric_sentences)}")

    # Ecriture fichier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = safe_output_filename(txt_path)

    output_text = build_output_text(
        doc_title=doc["title"] or txt_path.stem,
        ideas=final_ideas,
        numeric_sentences=numeric_sentences
    )
    out_path.write_text(output_text, encoding="utf-8")

    log(f"Fichier écrit : {out_path}")
    log(f"Fin traitement : {txt_path.name}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("Démarrage du script")

    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {TXT_DIR}")

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    log(f"Nombre de fichiers texte trouvés : {len(txt_files)}")

    if not txt_files:
        log("Aucun fichier texte trouvé")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_directory(OUTPUT_DIR)
    log("Répertoire de sortie vidé")

    for txt_path in txt_files:
        try:
            process_document(txt_path)
        except Exception as exc:
            log(f"ERREUR {txt_path.name} : {exc}")

    log("Fin du script")


if __name__ == "__main__":
    main()
