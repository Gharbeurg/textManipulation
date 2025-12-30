#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de pré-traitement (8 étapes) avec fichiers intermédiaires.
- Prend un fichier d'entrée (txt, html, pdf, docx) et produit un texte "prêt".
- Écrit un fichier intermédiaire à chaque étape pour pouvoir relancer après une erreur.
- Si un fichier intermédiaire existe déjà, l'étape est sautée (sauf --force).

Étapes :
1) Extraction du texte utile
2) Normalisation encodage (UTF-8) + petits nettoyages de caractères
3) Détection langue + filtrage simple (garder la langue dominante)
4) Mise en minuscules
5) Nettoyage espaces/lignes
6) Suppression du bruit (URLs/emails/téléphones/page x/y, etc.)
7) Corrections simples (mots coupés par césure PDF : "intel-\nligence" -> "intelligence")
8) Segmentation (découpe en paragraphes/phrases + marquage simple)

Dépendances OPTIONNELLES :
- HTML : beautifulsoup4
- PDF : pymupdf (fitz) OU pypdf
- DOCX: python-docx
- Langue : langdetect

Installation (exemples) :
  pip install beautifulsoup4 langdetect pymupdf python-docx
"""

# bibliothèques
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path
import os


#variables
INPUT_FILE = Path("C:/PYTHON/.entree/pneumo.html")
OUTPUT_FILE = Path("C:/PYTHON/.data/fichierTravail.txt")
WORKDIR = Path("C:/PYTHON/.travail")

FORCE_RECALCUL = False                             # True = recalculer toutes les étapes
TARGET_LANGUAGE = None                             # ex: "fr", "en", ou None

# =========================================================

# -------------------------
# Utils fichiers
# -------------------------

def clean_previous_outputs():
    # supprimer le fichier de sortie
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    # supprimer les fichiers intermédiaires
    if WORKDIR.exists():
        for f in WORKDIR.glob("*.txt"):
            if f.is_file():
                f.unlink()

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stage_path(stage_num: int, stage_name: str) -> Path:
    return WORKDIR / f"{stage_num:02d}_{stage_name}.txt"


def run_stage(stage_num, stage_name, func, input_text=None, input_file=None) -> str:
    out_path = stage_path(stage_num, stage_name)

    if out_path.exists() and not FORCE_RECALCUL:
        return read_text_file(out_path)

    try:
        if input_file:
            result = func(input_file)
        else:
            result = func(input_text or "")
    except Exception as e:
        print(f"\n❌ Erreur étape {stage_num} ({stage_name}) : {e}", file=sys.stderr)
        raise

    write_text_file(out_path, result)
    return result


# -------------------------
# Étape 1 : extraction texte
# -------------------------

def extract_text_any(path: Path) -> str:
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return read_text_file(path)

    if ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(read_text_file(path), "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text("\n")

    if ext == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)

    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    return read_text_file(path)


# -------------------------
# Étape 2 : normalisation caractères
# -------------------------

def normalize_chars(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    return re.sub(r"[\x00-\x1F]", "", text)


# -------------------------
# Étape 3 : langue dominante
# -------------------------

def filter_language(text: str) -> str:
    if not TARGET_LANGUAGE:
        return text

    from langdetect import detect
    blocks = text.split("\n\n")
    kept = []
    for b in blocks:
        try:
            if detect(b) == TARGET_LANGUAGE:
                kept.append(b)
        except Exception:
            pass
    return "\n\n".join(kept) if kept else text


# -------------------------
# Étape 4 : minuscules
# -------------------------

def to_lower(text: str) -> str:
    return text.lower()


# -------------------------
# Étape 5 : nettoyage espaces
# -------------------------

def clean_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -------------------------
# Étape 6 : suppression bruit
# -------------------------

def remove_noise(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b\S+@\S+\.\S+\b", "", text)
    return "\n".join(l for l in text.splitlines() if l.strip())


# -------------------------
# Étape 7 : corrections PDF
# -------------------------

def fix_hyphenation(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


# -------------------------
# Étape 8 : segmentation
# -------------------------

def segment(text: str) -> str:
    paragraphs = text.split("\n\n")
    result = []
    for p in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", p)
        result.append("\n".join(s.strip() for s in sentences if s.strip()))
    return "\n\n".join(result)


# -------------------------
# MAIN
# -------------------------

def main():
    print("{} - Début du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    clean_previous_outputs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("{} - Lecture du fichier d entree".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print("{} - Etape 1 : Extraction".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t1 = run_stage(1, "extract", extract_text_any, input_file=INPUT_FILE)
    print("{} - Etape 2 : Normalisation".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t2 = run_stage(2, "normalize", normalize_chars, t1)
    print("{} - Etape 3 : Choix du langage".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t3 = run_stage(3, "language", filter_language, t2)
    print("{} - Etape 4 : Passage en minuscule".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t4 = run_stage(4, "lowercase", to_lower, t3)
    print("{} - Etape 5 : Traitement des espaces".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t5 = run_stage(5, "spaces", clean_spaces, t4)
    print("{} - Etape 6 : Réduction du bruit".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t6 = run_stage(6, "noise", remove_noise, t5)
    print("{} - Etape 7 : hyphénation".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t7 = run_stage(7, "hyphenation", fix_hyphenation, t6)
    print("{} - Etape 8 : Segmentation".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    t8 = run_stage(8, "segment", segment, t7)

    print("{} - Ecriture du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    write_text_file(OUTPUT_FILE, t8)
    print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if __name__ == "__main__":
    main()
