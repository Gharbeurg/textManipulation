#Lecture un fichier en entrée, suppression de toutes les lignes vides ou en doublon

# bibliothèques
from __future__ import annotations

import re
from pathlib import Path
from typing import List
from datetime import datetime

# Variables fichiers entrée / sortie
# variables
fichier_entree = "D:/CODING/.params/entree.txt"
fichier_sortie = "D:/CODING/.data/entree_ss_doublon.txt"

# URL simple (http/https)
URL_RE = re.compile(r"https?://[^\s]+")


def is_blank_line(line: str) -> bool:
    """Vrai si la ligne est vide ou ne contient que des espaces/tabulations."""
    return line.strip() == ""


def extract_urls(line: str) -> List[str]:
    """Retourne la liste des URLs trouvées dans la ligne."""
    return URL_RE.findall(line)


def is_url_only_line(line: str) -> bool:
    """
    Vrai si la ligne contient uniquement des URLs (séparées par des espaces),
    sans autre texte.
    """
    stripped = line.strip()
    if stripped == "":
        return False

    urls = extract_urls(stripped)
    if not urls:
        return False

    # On enlève toutes les URLs puis on vérifie qu'il ne reste rien (à part des espaces)
    leftover = URL_RE.sub("", stripped).strip()
    return leftover == ""


def normalize_url_only_line(line: str) -> str:
    """
    Pour une ligne "URLs uniquement", enlève les URLs en doublon DANS la ligne,
    en gardant l'ordre d'apparition.
    """
    urls = extract_urls(line.strip())
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return " ".join(unique_urls) + "\n"


def process_file(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    seen_lines = set()

    print("{} - Ouverture du fichier d entrée".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with input_path.open("r", encoding="utf-8", errors="replace") as fin, \
         output_path.open("w", encoding="utf-8", newline="") as fout:

        for raw_line in fin:
            # 1) Supprimer lignes vides / espaces
            if is_blank_line(raw_line):
                continue

            line_to_write = raw_line

            # 2) Si ligne "URLs uniquement", supprimer les doublons d'URLs dans la ligne
            if is_url_only_line(raw_line):
                line_to_write = normalize_url_only_line(raw_line)

                # Si ça devient vide (ex: ligne bizarre), on skip
                if is_blank_line(line_to_write):
                    continue

            # 3) Supprimer lignes en doublon (après normalisation éventuelle)
            # On compare sur le texte exact (incluant ou non le \n final, ici on le garde)
            if line_to_write in seen_lines:
                continue

            seen_lines.add(line_to_write)
            fout.write(line_to_write)


def main() -> None:

    print("{} - Début du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    process_file(fichier_entree, fichier_sortie)
    print("{} - Ecriture du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))


if __name__ == "__main__":
    main()

print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
