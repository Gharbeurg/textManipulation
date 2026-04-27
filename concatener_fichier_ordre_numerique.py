import os
import re
from pathlib import Path

# === PARAMÈTRE ===
# Répertoire contenant les fichiers texte
repertoire = Path(r"C:/PYTHON/.data/ResultatsMarkdown")

# Nom du fichier de sortie
fichier_sortie = os.path.join(repertoire, "concatene.txt")

# Ligne de séparation entre les fichiers
separateur = "\n" + "=" * 50 + "\n"


def extraire_numero(nom_fichier):
    """
    Extrait le dernier nombre présent dans le nom du fichier.
    Exemple :
    - 'monfichier-2.md'  -> 2
    - 'monfichier-10.md' -> 10

    Si aucun nombre n'est trouvé, retourne un très grand nombre
    pour placer le fichier à la fin.
    """
    match = re.search(r'(\d+)(?=\.[^.]+$)', nom_fichier)
    if match:
        return int(match.group(1))
    return float('inf')


def concatener_fichiers_md(repertoire, fichier_sortie):
    # Liste des fichiers .md
    fichiers = [f for f in os.listdir(repertoire) if f.endswith(".md")]

    # Tri numérique selon le numéro dans le nom du fichier
    fichiers = sorted(fichiers, key=extraire_numero)

    print("Ordre de concaténation :")
    for fichier in fichiers:
        print(f" - {fichier}")

    with open(fichier_sortie, "w", encoding="utf-8") as sortie:
        for i, fichier in enumerate(fichiers):
            chemin_fichier = os.path.join(repertoire, fichier)

            with open(chemin_fichier, "r", encoding="utf-8") as f:
                contenu = f.read()

            sortie.write(contenu)

            # Ajouter séparateur sauf après le dernier fichier
            if i < len(fichiers) - 1:
                sortie.write(separateur)

    print(f"\nFichier généré : {fichier_sortie}")


# Exécution
concatener_fichiers_md(repertoire, fichier_sortie)