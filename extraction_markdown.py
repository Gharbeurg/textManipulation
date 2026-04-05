# BIBLIOTHEQUES
import requests
import shutil
from pathlib import Path
from datetime import datetime

# =========================
# PARAMETRES
# =========================

SOURCE_DIR  = "C:/PYTHON/.entree/SourcesSYNTHESE"   # Répertoire source des fichiers .md
OUTPUT_DIR  = "C:/PYTHON/.data/ResultatsIdees"       # Répertoire de sortie (vidé au démarrage)

# MODEL_NAME  = "qwen2.5:14b"
MODEL_NAME  = "qwen3.5:latest"                             # Modèle Ollama
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"      # URL de l'API Ollama

CHUNK_SIZE_WORDS = 1000   # [OPT 3] Taille cible des chunks en mots (500 → 1000 = 2× moins de chunks)
NUM_CTX          = 8192   # Taille du contexte LLM
TEMPERATURE      = 0.1    # Faible pour rester factuel
NUM_PREDICT      = 4096   # Tokens max par réponse (augmenté car les appels fusionnés retournent plus)


# =========================
# UTILITAIRES
# =========================

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def clear_output_dir(output_dir: str) -> None:
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] Output directory cleared: {output_dir}")


def get_markdown_files(source_dir: str) -> list:
    return sorted(Path(source_dir).glob("*.md"))


def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS) -> list:
    """Split text into chunks of ~chunk_size_words words, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks = []
    current_buf = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # Paragraph too large on its own: flush buffer then split by words
        if para_words > chunk_size_words:
            if current_buf:
                chunks.append("\n\n".join(current_buf))
                current_buf, current_words = [], 0
            words = para.split()
            for i in range(0, len(words), chunk_size_words):
                chunks.append(" ".join(words[i : i + chunk_size_words]))
            continue

        if current_words + para_words > chunk_size_words and current_buf:
            chunks.append("\n\n".join(current_buf))
            current_buf, current_words = [], 0

        current_buf.append(para)
        current_words += para_words

    if current_buf:
        chunks.append("\n\n".join(current_buf))

    return chunks


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "think": False,          # Désactive le thinking Qwen3 (30-50% plus rapide)
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return ((data.get("message") or {}).get("content") or "").strip()


def parse_bullet_list(text: str) -> list:
    """Extract items from a bullet list (lines starting with '- ')."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") and len(line) > 2:
            items.append(line[2:].strip())
    return items


def parse_dual_output(text: str) -> tuple:
    """Parse a response containing ### NUMERIC DATA and ### IDEAS sections."""
    numeric = []
    ideas   = []
    current_section = None

    for line in text.splitlines():
        stripped = line.strip()
        upper    = stripped.upper()
        if "NUMERIC DATA" in upper:
            current_section = "numeric"
        elif "IDEAS" in upper and stripped.startswith("#"):
            current_section = "ideas"
        elif stripped.startswith("- ") and len(stripped) > 2:
            item = stripped[2:].strip()
            if current_section == "numeric":
                numeric.append(item)
            elif current_section == "ideas":
                ideas.append(item)

    return numeric, ideas


# =========================
# [OPT 1] EXTRACTION FUSIONNÉE — LLM 1+3
# =========================

def extract_numeric_and_ideas(chunks: list) -> tuple:
    """[OPT 1] Single LLM call per chunk: extract numeric sentences AND ideas simultaneously."""
    all_numeric = []
    all_ideas   = []

    for i, chunk in enumerate(chunks, 1):
        prompt = f"""You are analyzing a text excerpt. Perform TWO extractions simultaneously.

### TASK 1 — NUMERIC DATA
Extract ONLY sentences containing meaningful numeric data.

INCLUDE:
- Statistics and percentages (e.g., "80% of cases", "increased by 12%")
- Measurements and quantities (e.g., "3.5 million euros", "over 200 countries")
- Comparisons involving numbers (e.g., "+15% compared to last year")
- Significant counts or rankings with explanatory context

EXCLUDE (noise):
- Page numbers and footnote references
- Dates used only as time markers (e.g., "In 2020, the company was founded")
- Version numbers, IDs, reference codes
- Simple list numbering (1. 2. 3.)
- Numbers without meaningful context

### TASK 2 — IDEAS
Extract ALL significant ideas, arguments, findings, and conclusions.

Rules:
- Each idea must be self-contained (understandable without the original text)
- Be concise but precise: one sentence per idea
- Cover all important points: main arguments, conclusions, recommendations, key findings
- Do not invent or infer beyond what is explicitly written

---
Return your answer using EXACTLY this format (keep the section headers):

### NUMERIC DATA
- sentence with numeric data
(write NONE if no relevant sentences)

### IDEAS
- extracted idea
(write NONE if no significant ideas)

TEXT EXCERPT {i}/{len(chunks)}:
{chunk}
"""
        result = call_ollama(prompt)
        numeric, ideas = parse_dual_output(result)
        all_numeric.extend(numeric)
        all_ideas.extend(ideas)

    return all_numeric, all_ideas


# =========================
# VERIFICATION — LLM 2A (nettoyage global)
# =========================

def clean_numeric_data(extracted: list) -> list:
    """LLM 2A — Global pass: remove noise from the extracted numeric sentences list."""
    if not extracted:
        return []

    extracted_text = "\n".join(f"- {s}" for s in extracted)
    prompt = f"""Review the following list of sentences extracted as containing meaningful numeric data.

Your task:
- REMOVE any sentence that is noise (page number, simple date reference, ID, version number, list numbering, number without context)
- KEEP all sentences with real statistics, measurements, percentages, or significant quantities

EXTRACTED SENTENCES:
{extracted_text}

Return ONLY the cleaned bullet list, one per line starting with "- ".
If the list becomes empty, return exactly: NONE
"""
    result = call_ollama(prompt)
    if result.strip().upper() == "NONE":
        return []
    return parse_bullet_list(result)


# =========================
# [OPT 2] VERIFICATION FUSIONNÉE — LLM 2B+4
# =========================

def verify_missing(chunks: list, numeric_cleaned: list, ideas_extracted: list) -> tuple:
    """[OPT 2] Single LLM call per chunk: find missing numeric sentences AND missing ideas simultaneously."""
    numeric_combined = list(numeric_cleaned)
    ideas_combined   = list(ideas_extracted)

    for i, chunk in enumerate(chunks, 1):
        numeric_text = "\n".join(f"- {s}" for s in numeric_combined) if numeric_combined else "NONE"
        ideas_text   = "\n".join(f"- {s}" for s in ideas_combined)   if ideas_combined   else "NONE"

        prompt = f"""You are a quality controller. Check this text excerpt for TWO types of missing content.

### TASK 1 — MISSING NUMERIC DATA
Check if the excerpt contains sentences with meaningful numeric data NOT yet in the list below.
Meaningful: statistics, percentages, measurements, comparisons, significant quantities.
NOT noise: page numbers, dates, IDs, version numbers, list numbers.

ALREADY EXTRACTED NUMERIC SENTENCES:
{numeric_text}

### TASK 2 — MISSING IDEAS
Check if the excerpt contains important ideas NOT yet captured in the list below.
- ADD only genuinely missing ideas
- Do NOT duplicate ideas already in the list (even if worded differently)
- Do NOT invent or infer beyond what is explicitly written

ALREADY EXTRACTED IDEAS:
{ideas_text}

---
Return your answer using EXACTLY this format (keep the section headers):

### MISSING NUMERIC DATA
- missing sentence
(write NONE if nothing is missing)

### MISSING IDEAS
- missing idea
(write NONE if nothing is missing)

TEXT EXCERPT {i}/{len(chunks)}:
{chunk}
"""
        result = call_ollama(prompt)
        missing_numeric, missing_ideas = parse_dual_output(result)

        for item in missing_numeric:
            if item not in numeric_combined:
                numeric_combined.append(item)
        for idea in missing_ideas:
            if idea not in ideas_combined:
                ideas_combined.append(idea)

    return numeric_combined, ideas_combined


# =========================
# SORTIE
# =========================

def save_output(source_name: str, numeric_data: list, ideas: list) -> None:
    stem = Path(source_name).stem
    output_path = Path(OUTPUT_DIR) / f"{stem}_extracted.md"

    lines = [f"# Source: {source_name}", ""]

    lines.append("## Numeric Data")
    if numeric_data:
        for item in numeric_data:
            lines.append(f"- {item}")
    else:
        lines.append("*No significant numeric data found.*")

    lines.append("")
    lines.append("## Ideas")
    if ideas:
        for idea in ideas:
            lines.append(f"- {idea}")
    else:
        lines.append("*No ideas found.*")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# TRAITEMENT D'UN FICHIER
# =========================

def process_file(filepath) -> bool:
    """Orchestrate the full processing of one markdown file. Returns True on success."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            print(f"  [SKIP] Empty file.")
            return True

        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunk(s)")

        # [OPT 1] Single call per chunk for both extractions
        print(f"  -> [LLM 1+3] Extracting numeric data and ideas...")
        numeric_data, ideas = extract_numeric_and_ideas(chunks)
        print(f"     {len(numeric_data)} numeric sentence(s), {len(ideas)} idea(s) extracted")

        # Noise cleaning (global, single call)
        print(f"  -> [LLM 2A] Cleaning numeric data (removing noise)...")
        numeric_data = clean_numeric_data(numeric_data)
        print(f"     {len(numeric_data)} sentence(s) after cleaning")

        # [OPT 2] Single call per chunk for both verifications
        print(f"  -> [LLM 2B+4] Verifying completeness (numeric + ideas)...")
        numeric_data, ideas = verify_missing(chunks, numeric_data, ideas)
        print(f"     {len(numeric_data)} numeric sentence(s), {len(ideas)} idea(s) after verification")

        save_output(filepath.name, numeric_data, ideas)
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


# =========================
# MAIN
# =========================

def main():
    print(f"[{ts()}] Starting extraction")

    clear_output_dir(OUTPUT_DIR)

    files = get_markdown_files(SOURCE_DIR)
    if not files:
        print(f"[{ts()}] No markdown files found in: {SOURCE_DIR}")
        return

    total = len(files)
    errors = 0
    print(f"[{ts()}] {total} file(s) to process\n")

    for idx, filepath in enumerate(files, 1):
        print(f"[{ts()}] [{idx}/{total}] {filepath.name}")
        success = process_file(filepath)
        if not success:
            errors += 1
        print()

    print(f"[{ts()}] Done. {total - errors}/{total} file(s) processed successfully.")
    if errors:
        print(f"[{ts()}] {errors} file(s) failed — check errors above.")


if __name__ == "__main__":
    main()
