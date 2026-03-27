# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
from typing import List, Dict
import re
import time

import requests


# ============================================================
# PARAMETRES
# ============================================================

TXT_DIR = Path(r"C:/PYTHON/.entree/Sources")
OUTPUT_DIR = Path(r"C:/PYTHON/.data/ResultatsIdees")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:8b"

OUTPUT_LANGUAGE = "English"

TARGET_CHUNK_WORDS = 1200
MAX_CHUNK_WORDS = 1600
MIN_CHUNK_WORDS = 350

REQUEST_TIMEOUT = 240
MAX_NUMERIC_SENTENCES = 200


# ============================================================
# LOG
# ============================================================

def log(label: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {label}", flush=True)


# ============================================================
# OUTILS TEXTE
# ============================================================

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
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
        \b\d+(?:[.,]\d+)?\s*(?:%|‰)
        |
        \b\d+(?:[.,]\d+)?\s*(?:percent|percentage points?)\b
        |
        \b\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?
        |
        \b\d+(?:[.,]\d+)?
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
)

USEFUL_QUANT_PATTERNS = [
    r"\bpercent\b",
    r"\bpercentage points?\b",
    r"\bincrease(?:d)?\b",
    r"\bdecrease(?:d)?\b",
    r"\bgrowth\b",
    r"\bdecline\b",
    r"\breduction\b",
    r"\bimprov(?:e|ed|ement)\b",
    r"\baverage\b",
    r"\btotal\b",
    r"\bshare\b",
    r"\brate\b",
    r"\bmore than\b",
    r"\bless than\b",
    r"\bup to\b",
    r"\bby 20\d{2}\b",
    r"\bbetween 20\d{2} and 20\d{2}\b",
    r"\bworkers?\b",
    r"\bemployees?\b",
    r"\bcompanies?\b",
    r"\barticles?\b",
    r"\bhours?\b",
    r"\bcalls?\b",
    r"\bjobs?\b",
    r"\bskills?\b",
    r"\boccupations?\b",
    r"\btraining\b",
    r"\brevenue\b",
    r"\bcost\b",
    r"\bwage\b",
    r"\bproductivity\b",
]

NOISE_QUANT_PATTERNS = [
    r"\bpage\b",
    r"\bchapter\b",
    r"\bsection\b",
    r"\btable\b",
    r"\bfigure\b",
    r"\bfig\.\b",
    r"\bexhibit\b",
    r"\bnote\b",
    r"\bappendix\b",
    r"\bdoi\b",
    r"\bissn\b",
    r"\bdownloaded from\b",
    r"\bcopyright\b",
    r"\blicense\b",
    r"\bopen access\b",
    r"\bjournal\b",
    r"\bvol(?:ume)?\b",
    r"\bissue\b",
    r"https?://",
    r"\buniversit(y|ies)\b",
    r"\bdepartment\b",
    r"\bemail\b",
]

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


def contains_number_data(text: str) -> bool:
    return bool(NUMBER_PATTERN.search(text))


def has_useful_quant_marker(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in USEFUL_QUANT_PATTERNS)


def has_noise_quant_marker(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in NOISE_QUANT_PATTERNS)


def split_into_sentences_numeric(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"\bet al\.", "et al", text, flags=re.IGNORECASE)
    text = re.sub(r"\be\.g\.", "eg", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.", "ie", text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÀ-ÖØ-Ý0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def clean_numeric_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)
    sentence = re.sub(r"\s+\)", ")", sentence)
    sentence = re.sub(r"\(\s+", "(", sentence)
    return sentence.strip(" -•\t")


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

    if ";" in sentence and low.count("(") >= 2:
        return True

    if re.search(r"\b[a-zà-ÿ-]+ et al\b", low):
        return True

    if low.count(",") > 6:
        return True

    return False


def looks_like_layout_fragment(text: str) -> bool:
    s = text.strip()

    if len(s) < 35:
        return True

    if s.count("\n") >= 2:
        return True

    if re.fullmatch(r"[\d\s,.;:()/%\-]+", s):
        return True

    if re.fullmatch(r"\d{1,4}", s):
        return True

    if re.search(r"\b\d+\s*/\s*\d+\b", s):
        return True

    return False


def looks_like_broken_sentence(text: str) -> bool:
    s = text.strip()

    if s.count("|") >= 2:
        return True

    if s.count(" - ") >= 3:
        return True

    if not re.search(
        r"\b(is|are|was|were|be|been|has|have|had|shows|showed|found|finds|reported|grew|rose|fell|cut|reduced|increased|improved|yielded|accounted|represented|reached|dropped)\b",
        s.lower()
    ):
        if not (re.search(r"%|percent|percentage points?", s.lower()) and has_useful_quant_marker(s)):
            return True

    return False


def score_numeric_sentence(sentence: str) -> int:
    s = sentence.strip()
    low = s.lower()
    score = 0

    if re.search(r"%|percent|percentage points?", low):
        score += 3

    if re.search(r"\b(increase|increased|decrease|decreased|growth|decline|reduction|improved|improvement|rose|fell|cut|dropped|yielded|reached)\b", low):
        score += 3

    if re.search(r"\b(workers|employees|companies|articles|hours|calls|jobs|skills|occupations|revenue|cost|wage|productivity)\b", low):
        score += 2

    if re.search(r"\b(by 20\d{2}|between 20\d{2} and 20\d{2})\b", low):
        score += 1

    if has_noise_quant_marker(s):
        score -= 4

    if looks_like_layout_fragment(s):
        score -= 4

    if looks_like_broken_sentence(s):
        score -= 3

    return score


def extract_numeric_sentences(text: str) -> List[str]:
    log("Extraction des phrases avec données chiffrées")

    sentences = split_into_sentences_numeric(text)
    kept: List[Dict] = []

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
        if looks_like_reference_fragment(s):
            continue
        if not has_useful_quant_marker(s):
            continue

        score = score_numeric_sentence(s)
        if score < 2:
            continue

        kept.append({
            "sentence": s,
            "score": score
        })

    best_by_key: Dict[str, Dict] = {}
    for item in kept:
        key = re.sub(r"\s+", " ", item["sentence"].strip()).lower()
        current = best_by_key.get(key)
        if current is None or item["score"] > current["score"]:
            best_by_key[key] = item

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda x: (x["score"], len(x["sentence"])), reverse=True)

    return [x["sentence"] for x in deduped[:MAX_NUMERIC_SENTENCES]]


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

    for txt_path in txt_files:
        try:
            process_document(txt_path)
        except Exception as exc:
            log(f"ERREUR {txt_path.name} : {exc}")

    log("Fin du script")


if __name__ == "__main__":
    main()