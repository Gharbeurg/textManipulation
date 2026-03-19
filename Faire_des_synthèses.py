from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from datetime import datetime

import requests

TXT_DIR = Path(r"C:/PYTHON/.entree/Sources")
OUTPUT_FILE = Path(r"C:/PYTHON/.data/Synthèse.txt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"

REQUEST_TIMEOUT = 600
SEPARATOR = "\n--------------------------------------\n"


def log(message: str):
    print("{} - {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), message))


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    log("Appel Ollama...")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    text = data.get("response", "").strip()

    if not text:
        raise RuntimeError("Réponse vide de Ollama.")

    log("Réponse Ollama reçue")
    return text


def read_text_file(file_path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Impossible de lire le fichier : {file_path}")


def build_ideas_prompt(title: str, content: str) -> str:
    return f"""
Extrais toutes les idées importantes du document.
Réponds en français.
Une idée par ligne.
Ne rien inventer.

Titre : {title}

Document :
\"\"\"
{content}
\"\"\"
""".strip()


def build_summary_prompt(title: str, content: str) -> str:
    return f"""
Fais une synthèse fidèle en français.
Conserve toutes les idées importantes.
Ne rien inventer.

Titre : {title}

Document :
\"\"\"
{content}
\"\"\"
""".strip()


def build_check_prompt(title: str, ideas: str, summary: str) -> str:
    return f"""
Vérifie si la synthèse contient toutes les idées.

Titre : {title}

Idées :
\"\"\"
{ideas}
\"\"\"

Synthèse :
\"\"\"
{summary}
\"\"\"

Si rien ne manque, réponds exactement :
AUCUNE_IDEE_MANQUANTE

Sinon, donne seulement les idées manquantes, une par ligne.
""".strip()


def build_merge_prompt(title: str, summary: str, missing: str) -> str:
    return f"""
Réécris la synthèse en intégrant toutes les idées manquantes.

Titre : {title}

Synthèse actuelle :
\"\"\"
{summary}
\"\"\"

Idées manquantes :
\"\"\"
{missing}
\"\"\"

Réponds uniquement avec la synthèse finale.
""".strip()


def process_file(file_path: Path) -> str:
    log(f"Lecture du fichier : {file_path}")
    content = read_text_file(file_path)

    if not content:
        log(f"Fichier vide : {file_path.name}")
        return ""

    title = file_path.stem

    log(f"Extraction des idées : {file_path.name}")
    ideas = call_ollama(build_ideas_prompt(title, content))

    log(f"Résumé : {file_path.name}")
    summary = call_ollama(build_summary_prompt(title, content))

    log(f"Vérification des idées manquantes : {file_path.name}")
    missing = call_ollama(build_check_prompt(title, ideas, summary))

    if missing.strip() != "AUCUNE_IDEE_MANQUANTE":
        log(f"Idées manquantes détectées : {file_path.name}")
        summary = call_ollama(build_merge_prompt(title, summary, missing))
    else:
        log(f"Aucune idée manquante : {file_path.name}")

    return f"{title}\n{summary.strip()}\n"


def main() -> None:
    log("Démarrage du script")

    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {TXT_DIR}")

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    log(f"Nombre de fichiers trouvés : {len(txt_files)}")

    if not txt_files:
        log("Aucun fichier .txt trouvé")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = []

    for file_path in txt_files:
        try:
            log(f"Début traitement fichier : {file_path.name}")
            block = process_file(file_path)
            if block:
                blocks.append(block)
            log(f"Fin traitement fichier : {file_path.name}")
        except Exception as exc:
            log(f"ERREUR {file_path.name} : {exc}")

    log("Assemblage du fichier final")
    final_text = SEPARATOR.join(block.strip() for block in blocks if block.strip())

    OUTPUT_FILE.write_text(final_text + "\n", encoding="utf-8")

    log(f"Fichier de sortie créé : {OUTPUT_FILE}")
    log("Fin du script")


if __name__ == "__main__":
    main()