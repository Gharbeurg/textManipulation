# -*- coding: utf-8 -*-
"""
Convertisseur multi-format -> Markdown avec Docling
Formats supportés : PDF, DOCX, PPTX, XLSX, HTML, images, AsciiDoc, Markdown, CSV
Usage : python convert_to_markdown.py
Installation : pip install docling
"""

from pathlib import Path
from datetime import datetime

import logging
from docling.document_converter import DocumentConverter


INPUT_DIR  = Path(r"C:/PYTHON/.entree/SourcesDocs")
OUTPUT_DIR = Path(r"C:/PYTHON/.data/ResultatsMarkdown")

# Extensions reconnues par Docling et le suffixe InputFormat correspondant
SUPPORTED_EXTENSIONS = {
    # Documents Office
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    # Images (Docling applique OCR + layout)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp",
    # Web / texte structuré
    ".html", ".htm",
    ".md",
    ".adoc", ".asciidoc",
    ".csv",
}


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")


def clear_output_dir(directory: Path) -> None:
    """Supprime tous les fichiers du répertoire de sortie (sans supprimer le répertoire)."""
    if not directory.exists():
        log(f"Répertoire de sortie inexistant, il sera créé à la première conversion.")
        return

    files = [f for f in directory.iterdir() if f.is_file()]
    if not files:
        log("Répertoire de sortie déjà vide.")
        return

    for f in files:
        f.unlink()
    log(f"Répertoire de sortie vidé ({len(files)} fichier(s) supprimé(s)).")


def collect_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """
    Parcourt le répertoire (non récursif) et sépare :
    - les fichiers à convertir (extension supportée)
    - les fichiers ignorés (extension inconnue)
    """
    supported, ignored = [], []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(f)
        else:
            ignored.append(f)
    return supported, ignored


def convert_file(path: Path, out_dir: Path, converter) -> bool:
    """Convertit un fichier en Markdown. Retourne True si succès."""
    try:
        log(f"Conversion [{path.suffix.upper()}] : {path.name}")
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.md"
        out_path.write_text(markdown, encoding="utf-8")

        log(f"  -> Sauvegardé : {out_path.name}")
        return True

    except Exception as e:
        log(f"  !! ERREUR ({path.name}) : {type(e).__name__} : {e}")
        return False


def main() -> None:
    
    logging.getLogger("docling").setLevel(logging.ERROR)

    log("START")
    clear_output_dir(OUTPUT_DIR)

    if not INPUT_DIR.exists():
        log(f"Répertoire source introuvable : {INPUT_DIR}")
        return

    supported, ignored = collect_files(INPUT_DIR)

    if ignored:
        log(f"{len(ignored)} fichier(s) ignoré(s) (format non supporté) :")
        for f in ignored:
            log(f"  - {f.name}")

    if not supported:
        log("Aucun fichier convertible trouvé.")
        return

    log(f"{len(supported)} fichier(s) à convertir.")

    converter = DocumentConverter()
    success = 0

    for path in supported:
        if convert_file(path, OUTPUT_DIR, converter):
            success += 1

    log(f"DONE - {success}/{len(supported)} fichier(s) converti(s) avec succès.")


if __name__ == "__main__":
    main()