from __future__ import annotations

import csv
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime


# =========================
# Variables à modifier
# =========================
INPUT_FILE = Path(r"C:/PYTHON/.params/entree.txt")
OUTPUT_CSV = Path(r"C:/PYTHON/.data/expressions_frequentes.csv")

# Groupes de mots à analyser (2 = bigrammes, 3 = trigrammes)
NGRAM_SIZES = (2, 3)

# Filtres / limites
MIN_TOKEN_LEN = 2          # ignore les mots très courts
MIN_COUNT = 2              # ignore les expressions trop rares
TOP_K = 200                # nombre max de lignes dans le CSV (par taille d'ngram)


# Stopwords (mots fréquents peu utiles) — petite liste FR, ajustable
FRENCH_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en",
    "et", "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me",
    "meme", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
    "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te",
    "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous",
    "d", "l", "c", "s", "n", "t", "y",
}


# =========================
# Lecture fichier (txt / docx / pdf)
# =========================
def read_text_file(path: Path) -> str:
    print("{} - Lecture du fichier d entree".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".log", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".docx":
        try:
            from docx import Document  # pip install python-docx
        except ImportError:
            raise SystemExit(
                "Fichier .docx détecté, mais 'python-docx' n'est pas installé.\n"
                "Installez-le avec: pip install python-docx"
            )
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # pip install pypdf
        except ImportError:
            raise SystemExit(
                "Fichier .pdf détecté, mais 'pypdf' n'est pas installé.\n"
                "Installez-le avec: pip install pypdf"
            )
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    # Par défaut, tente une lecture texte
    return path.read_text(encoding="utf-8", errors="replace")


# =========================
# Nettoyage / tokenisation
# =========================
def normalize_text(text: str) -> str:
    print("{} - Normalisation du texte".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    # minuscules
    text = text.lower()

    # remplace apostrophes typographiques par apostrophe simple
    text = text.replace("’", "'")

    # enlève la ponctuation (en gardant les apostrophes comme séparateurs)
    # ex: "l'équipe" -> "l équipe"
    text = re.sub(r"[’']", " ", text)

    # supprime ponctuation standard
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))

    # espace multiple -> espace simple
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    print("{} - Tokenisation".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    # garde lettres (y compris accents) et chiffres, en séparant sur le reste
    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüÿñæœ0-9]+", text, flags=re.IGNORECASE)
    cleaned = []
    for t in tokens:
        t = t.strip().lower()
        if len(t) < MIN_TOKEN_LEN:
            continue
        if t in FRENCH_STOPWORDS:
            continue
        cleaned.append(t)
    return cleaned


# =========================
# N-grammes
# =========================
def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def count_ngrams(tokens: List[str], n: int) -> Counter:
    c = Counter()
    for ng in ngrams(tokens, n):
        c[ng] += 1
    return c


# =========================
# Export CSV
# =========================
def write_csv(output_path: Path, results: List[dict]) -> None:
    print("{} - Ecriture du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ngram", "n", "count"])
        writer.writeheader()
        writer.writerows(results)


# =========================
# Main
# =========================
def main() -> None:
    print("{} - Début du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    if not INPUT_FILE.exists():
        raise SystemExit(f"Fichier introuvable: {INPUT_FILE}")

    raw_text = read_text_file(INPUT_FILE)
    norm = normalize_text(raw_text)
    tokens = tokenize(norm)

    all_rows: List[dict] = []

    for n in NGRAM_SIZES:
        counts = count_ngrams(tokens, n)
        # filtre + tri
        items = [(ng, ct) for ng, ct in counts.items() if ct >= MIN_COUNT]
        items.sort(key=lambda x: (-x[1], x[0]))

        for ng, ct in items[:TOP_K]:
            all_rows.append(
                {
                    "ngram": " ".join(ng),
                    "n": n,
                    "count": ct,
                }
            )

    write_csv(OUTPUT_CSV, all_rows)
    print("{} - Nombre de lignes du fichier de sortie : {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(all_rows)))

if __name__ == "__main__":
    main()

print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
