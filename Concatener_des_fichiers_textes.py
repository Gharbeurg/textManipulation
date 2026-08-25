import os
import re
from pathlib import Path


# === PARAMÈTRES ===

repertoire = Path(r"C:/PYTHON/.data/ResultatsMarkdown")

fichier_sortie = os.path.join(repertoire, "concatene.html")

separateur = "\n" + "=" * 50 + "\n"


def extraire_numero(nom_fichier):
    """
    Extrait le dernier nombre présent avant l'extension du fichier.

    Exemples :
    - monfichier-2.md  -> 2
    - monfichier-10.md -> 10

    Les fichiers sans numéro sont placés à la fin.
    """
    match = re.search(r"(\d+)(?=\.[^.]+$)", nom_fichier)

    if match:
        return int(match.group(1))

    return float("inf")


def lire_fichier_texte(chemin_fichier):
    """
    Lit un fichier en essayant plusieurs encodages courants.
    """
    encodages = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "cp1252",
        "latin-1",
    ]

    for encodage in encodages:
        try:
            with open(chemin_fichier, "r", encoding=encodage) as fichier:
                contenu = fichier.read()

            print(
                f"Encodage de {os.path.basename(chemin_fichier)} : "
                f"{encodage}"
            )

            return contenu

        except UnicodeDecodeError:
            continue

    raise UnicodeError(
        f"Impossible de lire le fichier : {chemin_fichier}"
    )


def concatener_fichiers_md(repertoire, fichier_sortie):
    repertoire = Path(repertoire)

    if not repertoire.exists():
        raise FileNotFoundError(
            f"Le répertoire n'existe pas : {repertoire}"
        )

    fichiers = [
        fichier
        for fichier in os.listdir(repertoire)
        if fichier.lower().endswith(".html")
    ]

    fichiers = sorted(fichiers, key=extraire_numero)

    if not fichiers:
        print("Aucun fichier Markdown trouvé.")
        return

    print("Ordre de concaténation :")

    for fichier in fichiers:
        print(f" - {fichier}")

    with open(
        fichier_sortie,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as sortie:

        for i, fichier in enumerate(fichiers):
            chemin_fichier = repertoire / fichier

            contenu = lire_fichier_texte(chemin_fichier)

            sortie.write(contenu)

            if i < len(fichiers) - 1:
                sortie.write(separateur)

    print(f"\nFichier généré : {fichier_sortie}")


concatener_fichiers_md(repertoire, fichier_sortie)