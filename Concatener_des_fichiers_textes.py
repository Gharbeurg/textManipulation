import os
from pathlib import Path

# === PARAMÈTRE ===
# Répertoire contenant les fichiers texte
repertoire = Path(r"C:/PYTHON/.data/ResultatsIdees")

# Nom du fichier de sortie
fichier_sortie = os.path.join(repertoire, "concatene.txt")

# Ligne de séparation entre les fichiers
separateur = "\n" + "="*50 + "\n"

def concatener_fichiers_txt(repertoire, fichier_sortie):
    # Liste des fichiers .txt
    fichiers = [f for f in os.listdir(repertoire) if f.endswith(".txt")]

    # Trier les fichiers (optionnel mais utile)
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