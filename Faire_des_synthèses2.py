# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import re
import time
from typing import List, Dict

import requests


# ============================================================
# PARAMETRES
# ============================================================

TXT_DIR = Path(r"C:/PYTHON/.data/ResultatsPDF")
OUTPUT_FILE = Path(r"C:/PYTHON/.data/syntheses.txt")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:8b"

OUTPUT_LANGUAGE = "English"

TOP_STRUCTURING_IDEAS = 10
TOP_COMPLEMENTARY_IDEAS = 10

TARGET_CHUNK_WORDS = 1200
MAX_CHUNK_WORDS = 1600
MIN_CHUNK_WORDS = 350

WRITE_FINAL_SUMMARY = True
WRITE_DOCUMENT_PLAN = True
WRITE_ALL_IDEAS = True
WRITE_NUMERIC_SENTENCES = True

REQUEST_TIMEOUT = 240
SEPARATOR = "\n--------------------------------------\n\n"


# ============================================================
# LOG
# ============================================================

def log(label: str) -> None:
    print("{} - {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), label), flush=True)


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


# ============================================================
# SORTIE FICHIER
# ============================================================

def init_output_file() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("", encoding="utf-8")


def append_output(text: str, add_separator: bool) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        if add_separator:
            f.write(SEPARATOR)
        f.write(text.strip() + "\n")


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
# PARSING DU TEXTE STRUCTURE
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


# ============================================================
# PLAN DU DOCUMENT
# ============================================================

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


def infer_chunk_type(title: str, parent_title: str = "") -> str:
    t = f"{parent_title} {title}".lower()

    if "abstract" in t or "summary" in t:
        return "abstract"
    if "introduction" in t or "background" in t or "literature" in t:
        return "context"
    if "method" in t or "survey" in t or "data" in t or "measuring" in t:
        return "method"
    if "result" in t or "impact" in t or "regression" in t or "descriptive" in t:
        return "results"
    if "discussion" in t:
        return "discussion"
    if "conclusion" in t:
        return "conclusion"

    return "body"


def build_section_chunks(doc: Dict) -> List[Dict]:
    chunks: List[Dict] = []

    if doc["abstract"]:
        chunks.append({
            "section_title": "Abstract",
            "subsection_title": "",
            "chunk_type": "abstract",
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
                    "chunk_type": infer_chunk_type(section_title),
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
                    "chunk_type": infer_chunk_type(subsection["title"], parent_title=section_title),
                    "text": ch
                })

    return chunks


# ============================================================
# PROMPTS
# ============================================================

def build_chunk_prompt(doc_title: str, plan: str, chunk: Dict) -> str:
    section_label = chunk["section_title"]
    if chunk["subsection_title"]:
        section_label += f" > {chunk['subsection_title']}"

    task_rules = {
        "abstract": "Prioritize the core research question, main finding, and main implication.",
        "context": "Prioritize the problem addressed, limits of prior literature, and what the document adds.",
        "method": "Prioritize the data, variables, categories, and method only if they help interpret the findings.",
        "results": "Prioritize findings, contrasts, mechanisms, and differences across groups.",
        "discussion": "Prioritize interpretation, implications, and limits.",
        "conclusion": "Prioritize final conclusions, implications, and recommendations.",
        "body": "Prioritize concrete findings, arguments, and implications."
    }

    rule = task_rules.get(chunk["chunk_type"], task_rules["body"])

    return f"""
You are extracting high-value ideas from an academic or professional document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

CURRENT SECTION:
{section_label}

CHUNK TYPE:
{chunk["chunk_type"]}

TASK:
Extract 5 to 8 ideas from the text below.

IMPORTANT RULES:
- Keep only ideas that are important for understanding the document.
- Be specific and concrete.
- Do not repeat the same idea with different wording.
- Ignore metadata, page layout, repeated running titles, references, biographies, and decorative figure labels.
- Mention methods only when they are useful for interpreting the findings.
- Keep quantitative results when they matter.
- Each idea must be a full sentence.
- Each idea must be self-contained.
- Avoid vague wording like "the article discusses" or "the text mentions".
- {rule}

RETURN FORMAT:
One idea per line, starting with "- "

TEXT:
{chunk["text"]}
""".strip()


def build_consolidation_prompt(doc_title: str, plan: str, ideas: List[str]) -> str:
    joined = "\n".join(f"- {x}" for x in ideas)

    return f"""
You are consolidating ideas extracted from a document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

TASK:
From the list below, produce a cleaned list of distinct ideas.

RULES:
- Keep only ideas that are important for understanding the document.
- Merge duplicates and near-duplicates.
- Preserve important numbers and contrasts.
- Remove ideas that are too generic or too weak.
- Keep 15 to 25 ideas.
- Each idea must be one full sentence.
- One idea per line, starting with "- "

IDEAS:
{joined}
""".strip()


def build_ranking_prompt(doc_title: str, plan: str, ideas: List[str], n: int, mode: str) -> str:
    joined = "\n".join(f"- {x}" for x in ideas)

    if mode == "structuring":
        description = "Select the most structuring ideas, meaning the ideas without which the document would lose its core meaning."
    else:
        description = "Select complementary but still important ideas that enrich understanding of the document without repeating the core ideas."

    return f"""
You are selecting the strongest ideas from a document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

TASK:
{description}

RULES:
- Return exactly {n} ideas.
- Keep ideas that are concrete, not generic.
- Do not repeat ideas with slightly different wording.
- Preserve important numbers, group differences, and implications when they matter.
- Each idea must be one full sentence.
- One idea per line, starting with "- "

IDEAS TO CHOOSE FROM:
{joined}
""".strip()


def build_final_summary_prompt(
    doc_title: str,
    plan: str,
    structuring_ideas: List[str],
    complementary_ideas: List[str]
) -> str:
    joined1 = "\n".join(f"- {x}" for x in structuring_ideas)
    joined2 = "\n".join(f"- {x}" for x in complementary_ideas)

    return f"""
You are writing a final synthesis of a document.

OUTPUT LANGUAGE: {OUTPUT_LANGUAGE}

DOCUMENT TITLE:
{doc_title or "Unknown title"}

DOCUMENT PLAN:
{plan}

MOST STRUCTURING IDEAS:
{joined1}

COMPLEMENTARY IDEAS:
{joined2}

TASK:
Write one final synthesis of the key ideas.

RULES:
- Write one coherent synthesis, not bullets.
- Keep the hierarchy of the document.
- Highlight the research question, the main findings, the contrasts, and the implications.
- Keep important methodological information only when it helps understand the results.
- Keep important quantitative information when it matters.
- Avoid generic filler.
- Length target: 250 to 450 words.
""".strip()


# ============================================================
# SORTIE LLM -> LISTE D'IDEES
# ============================================================

def parse_bullet_lines(text: str) -> List[str]:
    lines: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
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
# PHRASES AVEC DONNEES CHIFFREES
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
    r"\be-?mail\b",
    r"\bvolume\b",
    r"\bissue\b",
    r"\blicense\b",
    r"\bcopyright\b",
    r"\bcreativecommons\b",
    r"https?://",
]

MONTHS_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE
)


def contains_number_data(text: str) -> bool:
    return bool(NUMBER_PATTERN.search(text))


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


def extract_numeric_sentences(text: str) -> List[str]:
    log("Extraction des phrases avec données chiffrées")

    sentences = split_into_sentences_numeric(text)
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

    return deduplicate_ideas_simple(results)


# ============================================================
# PIPELINE DOCUMENT
# ============================================================

def process_document(txt_path: Path) -> str:
    log(f"Début traitement : {txt_path.name}")

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)

    doc = parse_structured_text(text)
    plan = build_document_plan(doc)
    chunks = build_section_chunks(doc)

    log(f"Plan reconstruit : {txt_path.name}")
    log(f"Nombre de chunks : {len(chunks)}")

    all_ideas: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        log(f"Extraction idées chunk {i}/{len(chunks)} - {txt_path.name}")
        prompt = build_chunk_prompt(doc["title"], plan, chunk)
        response = ollama_generate(prompt, temperature=0.2, num_predict=900)
        ideas = parse_bullet_lines(response)

        if not ideas:
            ideas = split_into_sentences_basic(response)

        all_ideas.extend(ideas)
        time.sleep(0.2)

    all_ideas = [clean_idea_text(x) for x in all_ideas if clean_idea_text(x)]
    all_ideas = deduplicate_ideas_simple(all_ideas)

    log(f"Nombre d'idées brutes : {len(all_ideas)}")

    log(f"Consolidation des idées : {txt_path.name}")
    consolidation_prompt = build_consolidation_prompt(doc["title"], plan, all_ideas)
    consolidated_text = ollama_generate(consolidation_prompt, temperature=0.1, num_predict=1200)
    consolidated_ideas = parse_bullet_lines(consolidated_text)

    if not consolidated_ideas:
        consolidated_ideas = all_ideas[:]

    consolidated_ideas = deduplicate_ideas_simple(consolidated_ideas)

    log(f"Nombre d'idées consolidées : {len(consolidated_ideas)}")

    log(f"Sélection des idées structurantes : {txt_path.name}")
    prompt_structuring = build_ranking_prompt(
        doc["title"], plan, consolidated_ideas, TOP_STRUCTURING_IDEAS, "structuring"
    )
    structuring_text = ollama_generate(prompt_structuring, temperature=0.1, num_predict=1000)
    structuring_ideas = parse_bullet_lines(structuring_text)
    structuring_ideas = deduplicate_ideas_simple(structuring_ideas)[:TOP_STRUCTURING_IDEAS]

    remaining = [x for x in consolidated_ideas if x not in structuring_ideas]

    log(f"Sélection des idées complémentaires : {txt_path.name}")
    prompt_complementary = build_ranking_prompt(
        doc["title"], plan, remaining, TOP_COMPLEMENTARY_IDEAS, "complementary"
    )
    complementary_text = ollama_generate(prompt_complementary, temperature=0.1, num_predict=1000)
    complementary_ideas = parse_bullet_lines(complementary_text)
    complementary_ideas = deduplicate_ideas_simple(complementary_ideas)[:TOP_COMPLEMENTARY_IDEAS]

    final_summary = ""
    if WRITE_FINAL_SUMMARY:
        log(f"Rédaction de la synthèse finale : {txt_path.name}")
        prompt_summary = build_final_summary_prompt(
            doc["title"], plan, structuring_ideas, complementary_ideas
        )
        final_summary = ollama_generate(prompt_summary, temperature=0.15, num_predict=1600).strip()

    numeric_sentences = extract_numeric_sentences(text)

    block = build_output_block(
        doc_title=doc["title"] or txt_path.stem,
        plan=plan,
        raw_ideas=all_ideas,
        structuring_ideas=structuring_ideas,
        complementary_ideas=complementary_ideas,
        final_summary=final_summary,
        numeric_sentences=numeric_sentences
    )

    log(f"Fin traitement : {txt_path.name}")
    return block


# ============================================================
# FORMAT SORTIE
# ============================================================

def build_output_block(
    doc_title: str,
    plan: str,
    raw_ideas: List[str],
    structuring_ideas: List[str],
    complementary_ideas: List[str],
    final_summary: str,
    numeric_sentences: List[str]
) -> str:
    lines: List[str] = []

    lines.append("TITRE DU DOCUMENT")
    lines.append(doc_title or "Unknown title")
    lines.append("")

    if WRITE_DOCUMENT_PLAN:
        lines.append("PLAN DU DOCUMENT")
        lines.append(plan)
        lines.append("")

    if WRITE_ALL_IDEAS:
        lines.append("ALL EXTRACTED IDEAS")
        if raw_ideas:
            for idea in raw_ideas:
                lines.append(f"- {idea}")
        else:
            lines.append("- No extracted idea.")
        lines.append("")

    lines.append("10 IDEES LES PLUS STRUCTURANTES")
    if structuring_ideas:
        for idea in structuring_ideas:
            lines.append(f"- {idea}")
    else:
        lines.append("- No structuring idea.")
    lines.append("")

    lines.append("10 IDEES COMPLEMENTAIRES IMPORTANTES")
    if complementary_ideas:
        for idea in complementary_ideas:
            lines.append(f"- {idea}")
    else:
        lines.append("- No complementary idea.")
    lines.append("")

    if WRITE_FINAL_SUMMARY:
        lines.append("SYNTHESE FINALE DES IDEES CLES")
        lines.append(final_summary if final_summary else "No final summary.")
        lines.append("")

    if WRITE_NUMERIC_SENTENCES:
        lines.append("SENTENCES WITH NUMERICAL DATA")
        if numeric_sentences:
            for s in numeric_sentences:
                lines.append(f"- {s}")
        else:
            lines.append("- No sentence with numerical data found.")
        lines.append("")

    return "\n".join(lines).strip()


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

    init_output_file()

    first_written = False

    for txt_path in txt_files:
        try:
            block = process_document(txt_path)

            if block.strip():
                append_output(block, add_separator=first_written)
                first_written = True

        except Exception as exc:
            log(f"ERREUR {txt_path.name} : {exc}")

    log(f"Fichier final : {OUTPUT_FILE}")
    log("Fin du script")


if __name__ == "__main__":
    main()