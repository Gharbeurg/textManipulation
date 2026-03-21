from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import requests

# ============================================================
# CONFIGURATION
# ============================================================

TXT_DIR = Path(r"C:/PYTHON/.entree/Sources")
OUTPUT_FILE = Path(r"C:/PYTHON/.data/Synthèse.txt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

REQUEST_TIMEOUT = 1800
SEPARATOR = "\n--------------------------------------\n"

SECTION_MAX_CHARS = 16000
SECTION_OVERLAP = 1200

MAX_EXTRACT_IDEAS_PER_CHUNK = 8
MAX_CHUNK_SYNTHESIS_IDEAS = 6
MAX_GROUP_SYNTHESIS_IDEAS = 10
MAX_FINAL_SYNTHESIS_IDEAS = 20
GROUP_SIZE = 5

# ============================================================
# OUTILS DE BASE
# ============================================================

def log(message: str) -> None:
    print("{} - {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), message), flush=True)


def read_text_file(file_path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Impossible de lire le fichier : {file_path}")


def init_output_file() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("", encoding="utf-8")


def append_output_block(text: str, add_separator: bool) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        if add_separator:
            f.write(SEPARATOR)
        f.write(text.strip() + "\n")


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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


def split_list_into_groups(items: List[str], group_size: int) -> List[List[str]]:
    groups = []
    for i in range(0, len(items), group_size):
        groups.append(items[i:i + group_size])
    return groups


# ============================================================
# APPEL OLLAMA
# ============================================================

def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    log("Appel Ollama")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 4096,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Erreur lors de l'appel à Ollama : {exc}") from exc

    data = response.json()
    text = data.get("response", "").strip()

    if not text:
        raise RuntimeError("Réponse vide de Ollama.")

    log("Réponse Ollama reçue")
    return text


# ============================================================
# PARSING DES LISTES A PUCES
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

    return dedupe_preserve_order(items)


# ============================================================
# NORMALISATION EN ANGLAIS
# ============================================================

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


def normalize_bullets_to_english(items: List[str]) -> List[str]:
    cleaned_items = [item.strip() for item in items if item.strip()]

    if not cleaned_items:
        return []

    raw_text = "\n".join([f"- {item}" for item in cleaned_items])
    prompt = build_force_english_prompt(raw_text)
    raw = call_ollama(prompt)
    english_items = parse_bullet_list(raw)

    if not english_items:
        return cleaned_items

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

CUT_FROM_PATTERNS = [
    r"\nreferences\s*:\s*\n",
    r"\nreferences\s*\n",
    r"\nauthor contributions\s*:\s*\n",
    r"\nfunding\s*:\s*\n",
    r"\ninstitutional review board statement\s*:\s*\n",
    r"\ninformed consent statement\s*:\s*\n",
    r"\ndata availability statement\s*:\s*\n",
    r"\nconflicts of interest\s*:\s*\n",
    r"\ndisclaimer/publisher[’']?s note\s*:\s*\n",
    r"\ndisclaimer\s*:\s*\n",
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


def cut_from_first_match(text: str, patterns: List[str]) -> str:
    lower_text = text.lower()
    cut_positions = []

    for pattern in patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if match:
            cut_positions.append(match.start())

    if cut_positions:
        return text[:min(cut_positions)].strip()

    return text.strip()


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


def clean_source_text(raw_text: str) -> str:
    log("Nettoyage du texte brut")

    text = normalize_spaces(raw_text)
    text = cut_from_first_match(text, CUT_FROM_PATTERNS)
    text = remove_editorial_header_lines(text)
    text = remove_noise_lines(text)

    text = re.sub(
        r"\nTable\s+\d+\..*?(?=\n(?:[A-Z][^\n]{0,120}:|\d+\.\s+[A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)*\n|\Z))",
        "\n",
        text,
        flags=re.DOTALL
    )
    text = re.sub(
        r"\nFigure\s+\d+\..*?(?=\n(?:[A-Z][^\n]{0,120}:|\d+\.\s+[A-Z]|[A-Z][a-z]+(?: [A-Z][a-z]+)*\n|\Z))",
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

        if re.fullmatch(r"[\d\s,.\-%()]+", stripped):
            continue

        if len(stripped) <= 3 and re.fullmatch(r"[\d.%,-]+", stripped):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = normalize_spaces(text)

    return text


# ============================================================
# DECOUPAGE PAR SECTIONS
# ============================================================

SECTION_TITLE_PATTERNS = [
    r"^\d+(\.\d+)*\.\s+.+$",
    r"^(abstract|introduction|literature review|research methodology|methodology|results|discussion|conclusions?|limitations?|future research directions?)\s*$",
]


def is_section_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    for pattern in SECTION_TITLE_PATTERNS:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True

    return False


def split_into_sections(text: str) -> List[dict]:
    lines = text.splitlines()

    sections = []
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

    if not found_title and sections:
        return sections

    return sections if sections else [{"title": "Document", "text": text.strip()}]


def split_large_text(text: str, chunk_size: int = SECTION_MAX_CHARS, overlap: int = SECTION_OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end

        if target_end < text_length:
            newline_pos = text.rfind("\n", start, target_end)
            space_pos = text.rfind(" ", start, target_end)

            if newline_pos > start + int(chunk_size * 0.6):
                end = newline_pos
            elif space_pos > start + int(chunk_size * 0.6):
                end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_section_chunks(clean_text: str) -> List[dict]:
    log("Découpage en sections")
    sections = split_into_sections(clean_text)

    chunks = []

    for section in sections:
        section_title = section["title"]
        section_text = section["text"]

        if len(section_text) <= SECTION_MAX_CHARS:
            chunks.append({
                "section_title": section_title,
                "text": section_text
            })
        else:
            sub_chunks = split_large_text(section_text)
            for i, sub_chunk in enumerate(sub_chunks, start=1):
                chunks.append({
                    "section_title": f"{section_title} [part {i}/{len(sub_chunks)}]",
                    "text": sub_chunk
                })

    return chunks


# ============================================================
# PROMPTS DE SYNTHÈSE HIÉRARCHIQUE
# ============================================================

def build_extract_prompt(doc_title: str, section_title: str, chunk_text: str) -> str:
    return f"""
You are reading an excerpt from a document.

Task:
extract the key ideas that are genuinely present in this excerpt.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- Do not invent anything.
- Do not include editorial metadata.
- Do not include DOIs, licenses, copyright notices, or conflicts of interest.
- Do not include bibliography titles.
- Do not present references to other articles as if they were ideas of this document.
- Keep only ideas useful for understanding this excerpt.
- Give at most {MAX_EXTRACT_IDEAS_PER_CHUNK} ideas for this excerpt.
- If the excerpt contains few ideas, output only the ideas that are really present.
- If the excerpt contains no useful idea, respond exactly:
- No useful idea in this excerpt.

Document title: {doc_title}
Section: {section_title}

Text:
\"\"\"
{chunk_text}
\"\"\"
""".strip()


def build_chunk_synthesis_prompt(doc_title: str, section_title: str, extracted_ideas: List[str]) -> str:
    ideas_text = "\n".join([f"- {idea}" for idea in extracted_ideas])

    return f"""
You receive ideas extracted from one excerpt of a document.

Task:
write a short, high-quality synthesis of the most important ideas from this excerpt.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- Merge overlapping ideas when useful.
- Keep the most important information only.
- Keep at most {MAX_CHUNK_SYNTHESIS_IDEAS} ideas.
- Cover, when present: objective, method, main findings, limitations, implications.
- Do not invent anything.

Document title: {doc_title}
Excerpt: {section_title}

Extracted ideas:
\"\"\"
{ideas_text}
\"\"\"
""".strip()


def build_group_synthesis_prompt(doc_title: str, group_index: int, chunk_summaries: List[List[str]]) -> str:
    blocks = []

    for i, summary in enumerate(chunk_summaries, start=1):
        blocks.append(f"Excerpt summary {i}:")
        if summary:
            blocks.extend([f"- {idea}" for idea in summary])
        else:
            blocks.append("- No useful idea.")
        blocks.append("")

    summaries_text = "\n".join(blocks).strip()

    return f"""
You receive several short excerpt summaries from the same document.

Task:
produce an intermediate synthesis for this group of excerpt summaries.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- Merge repeated ideas.
- Keep only the strongest and most informative ideas.
- Keep at most {MAX_GROUP_SYNTHESIS_IDEAS} ideas.
- Preserve important points that appear only once if they are central.
- Cover, when present: objective, method, sample/data, main findings, limitations, recommendations.
- Do not invent anything.

Document title: {doc_title}
Group number: {group_index}

Excerpt summaries:
\"\"\"
{summaries_text}
\"\"\"
""".strip()


def build_final_hierarchical_synthesis_prompt(doc_title: str, group_summaries: List[List[str]]) -> str:
    blocks = []

    for i, summary in enumerate(group_summaries, start=1):
        blocks.append(f"Intermediate summary {i}:")
        if summary:
            blocks.extend([f"- {idea}" for idea in summary])
        else:
            blocks.append("- No useful idea.")
        blocks.append("")

    summaries_text = "\n".join(blocks).strip()

    return f"""
You receive intermediate summaries built from different parts of a long document.

Task:
produce the final synthesis of the key ideas of the whole document.

Rules:
- Output in English only.
- Respond only with a bullet list.
- One idea per line starting with "- ".
- No JSON.
- No headings.
- No introduction.
- No conclusion.
- Keep only the most important ideas for understanding the whole document.
- Merge duplicates and very similar ideas.
- Keep at most {MAX_FINAL_SYNTHESIS_IDEAS} ideas.
- Ensure balanced coverage when present:
  - objective or topic
  - method
  - sample or data
  - main findings
  - limitations
  - implications or recommendations
- Do not invent anything.

Document title: {doc_title}

Intermediate summaries:
\"\"\"
{summaries_text}
\"\"\"
""".strip()


# ============================================================
# EXTRACTION ET SYNTHESE HIÉRARCHIQUE
# ============================================================

def extract_ideas_from_chunk(doc_title: str, section_title: str, chunk_text: str) -> List[str]:
    prompt = build_extract_prompt(doc_title, section_title, chunk_text)
    raw = call_ollama(prompt)
    ideas = parse_bullet_list(raw)
    ideas = normalize_bullets_to_english(ideas)

    if not ideas:
        return ["No useful idea in this excerpt."]

    return ideas


def synthesize_chunk_ideas(doc_title: str, section_title: str, extracted_ideas: List[str]) -> List[str]:
    useful = [x for x in extracted_ideas if x.strip() and x.strip().lower() != "no useful idea in this excerpt."]

    if not useful:
        return []

    prompt = build_chunk_synthesis_prompt(doc_title, section_title, useful)
    raw = call_ollama(prompt)
    ideas = parse_bullet_list(raw)
    ideas = normalize_bullets_to_english(ideas)

    if not ideas:
        return useful[:MAX_CHUNK_SYNTHESIS_IDEAS]

    return ideas[:MAX_CHUNK_SYNTHESIS_IDEAS]


def synthesize_group(doc_title: str, group_index: int, chunk_summaries: List[List[str]]) -> List[str]:
    useful = [summary for summary in chunk_summaries if summary]

    if not useful:
        return []

    prompt = build_group_synthesis_prompt(doc_title, group_index, useful)
    raw = call_ollama(prompt)
    ideas = parse_bullet_list(raw)
    ideas = normalize_bullets_to_english(ideas)

    if not ideas:
        flat = []
        for summary in useful:
            flat.extend(summary)
        return dedupe_preserve_order(flat)[:MAX_GROUP_SYNTHESIS_IDEAS]

    return ideas[:MAX_GROUP_SYNTHESIS_IDEAS]


def build_hierarchical_final_synthesis(doc_title: str, group_summaries: List[List[str]]) -> List[str]:
    useful = [summary for summary in group_summaries if summary]

    if not useful:
        return []

    prompt = build_final_hierarchical_synthesis_prompt(doc_title, useful)
    raw = call_ollama(prompt)
    ideas = parse_bullet_list(raw)
    ideas = normalize_bullets_to_english(ideas)

    if not ideas:
        flat = []
        for summary in useful:
            flat.extend(summary)
        return dedupe_preserve_order(flat)[:MAX_FINAL_SYNTHESIS_IDEAS]

    return ideas[:MAX_FINAL_SYNTHESIS_IDEAS]


def run_hierarchical_synthesis(doc_title: str, chunks: List[dict]) -> tuple[List[str], List[str]]:
    all_extracted_ideas: List[str] = []
    chunk_summaries: List[List[str]] = []

    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        log(f"Extraction des idées - {doc_title} - morceau {index}/{total} - {chunk['section_title']}")
        extracted = extract_ideas_from_chunk(doc_title, chunk["section_title"], chunk["text"])
        all_extracted_ideas.extend(extracted)

        log(f"Synthèse locale du morceau - {doc_title} - morceau {index}/{total}")
        chunk_summary = synthesize_chunk_ideas(doc_title, chunk["section_title"], extracted)
        chunk_summaries.append(chunk_summary)

    group_of_chunk_summaries = split_list_into_groups(chunk_summaries, GROUP_SIZE)

    intermediate_group_summaries: List[List[str]] = []
    for group_index, group in enumerate(group_of_chunk_summaries, start=1):
        log(f"Synthèse intermédiaire du groupe - {doc_title} - groupe {group_index}/{len(group_of_chunk_summaries)}")
        group_summary = synthesize_group(doc_title, group_index, group)
        intermediate_group_summaries.append(group_summary)

    log(f"Synthèse finale hiérarchique - {doc_title}")
    final_synthesis = build_hierarchical_final_synthesis(doc_title, intermediate_group_summaries)

    all_extracted_ideas = dedupe_preserve_order(
        [x for x in all_extracted_ideas if x.strip() and x.strip().lower() != "no useful idea in this excerpt."]
    )

    return all_extracted_ideas, final_synthesis


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

MONTHS_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE
)


def contains_number_data(text: str) -> bool:
    return bool(NUMBER_PATTERN.search(text))


def split_into_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)

    text = re.sub(r"\bet al\.", "et al", text, flags=re.IGNORECASE)
    text = re.sub(r"\be\.g\.", "eg", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.", "ie", text, flags=re.IGNORECASE)

    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÀ-ÖØ-Ý0-9])", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences


def clean_numeric_sentence(sentence: str) -> str:
    sentence = normalize_spaces(sentence)
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
        return f"{title}\n- Content empty after cleaning.\n"

    chunks = build_section_chunks(clean_content)
    log(f"Nombre de morceaux pour {file_path.name} : {len(chunks)}")

    all_ideas, final_key_ideas = run_hierarchical_synthesis(title, chunks)
    final_key_ideas = normalize_bullets_to_english(final_key_ideas)

    numeric_sentences = extract_numeric_sentences(clean_content)

    return format_output_block(
        title,
        all_ideas,
        final_key_ideas,
        numeric_sentences
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