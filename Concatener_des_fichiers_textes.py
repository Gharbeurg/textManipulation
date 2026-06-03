import os
from pathlib import Path

# === PARAMÈTRE ===
# Répertoire contenant les fichiers Markdown
repertoire = Path(r"C:/PYTHON/.data/ResultatsMarkdown")

# Nom du fichier de sortie
fichier_sortie = os.path.join(repertoire, "concatene.md")

# Ligne de séparation entre les fichiers
separateur = "\n" + "=" * 50 + "\n"


def supprimer_fichier_sortie_si_existe(fichier_sortie):
    """
    Supprime le fichier de sortie s'il existe déjà.
    Cela évite de le reprendre dans la concaténation.
    """
    if os.path.exists(fichier_sortie):
        os.remove(fichier_sortie)
        print(f"Ancien fichier supprimé : {fichier_sortie}")


def concatener_fichiers_txt(repertoire, fichier_sortie):
    # Supprimer le fichier de sortie s'il existe déjà
    supprimer_fichier_sortie_si_existe(fichier_sortie)

    # Liste des fichiers .md
    fichiers = [f for f in os.listdir(repertoire) if f.endswith(".md")]

    # Exclure explicitement le fichier de sortie de la liste, par sécurité
    nom_fichier_sortie = os.path.basename(fichier_sortie)
    fichiers = [f for f in fichiers if f != nom_fichier_sortie]

    # Trier les fichiers
    fichiers.sort()

    with open(fichier_sortie, "w", encoding="utf-8") as sortie:
        for i, fichier in enumerate(fichiers):
            chemin_fichier = os.path.join(repertoire, fichier)

            with open(chemin_fichier, "r", encoding="utf-8") as f:
                contenu = f.read()

            # Écriture du contenu
            sortie.write(contenu)

            # Ajouter séparateur sauf après le dernier fichier
            if i < len(fichiers) - 1:
                sortie.write(separateur)

    print(f"Fichier généré : {fichier_sortie}")


# Exécution
concatener_fichiers_txt(repertoire, fichier_sortie)