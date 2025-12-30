"""
Nuage de mots (word cloud) à partir d'un fichier texte.

Ajouts :
- exclusion des mots < 4 lettres
- vérification si le texte filtré est vide
- si le masque empêche de dessiner, on réessaie sans masque

Bibliothèques :
pip install wordcloud pillow numpy matplotlib
"""

from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ============================================================
# PARAMÈTRES À MODIFIER ICI
# ============================================================

INPUT_FILE = "C:/PYTHON/.params/entree.txt"
OUTPUT_FILE = "C:/PYTHON/.params/nuage_mots.png"

FORME = "carre"          # "carre", "rond", "ovale", "none"
LARGEUR = 1200
HAUTEUR = 800

NB_MOTS = 200
LONGUEUR_MIN = 4

COULEURS = "viridis"
FOND = "white"
PREVIEW = True


# ============================================================
# CODE
# ============================================================

def lire_texte(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def filtrer_mots_courts(texte: str, longueur_min: int) -> str:
    # \w inclut lettres/chiffres/_ ; on garde simple ici
    mots = re.findall(r"\b\w+\b", texte.lower())
    mots_filtres = [m for m in mots if len(m) >= longueur_min]
    return " ".join(mots_filtres)


def creer_masque(forme: str, largeur: int, hauteur: int):
    """
    WordCloud : BLANC = interdit, NOIR = autorisé
    """
    forme = forme.lower()

    if forme == "none":
        return None

    # Fond BLANC = interdit partout
    image = Image.new("L", (largeur, hauteur), 255)
    draw = ImageDraw.Draw(image)

    # Forme NOIRE = zone autorisée
    if forme == "carre":
        draw.rectangle([0, 0, largeur - 1, hauteur - 1], fill=0)

    elif forme == "rond":
        r = min(largeur, hauteur) // 2
        cx, cy = largeur // 2, hauteur // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)

    elif forme == "ovale":
        draw.ellipse([0, 0, largeur - 1, hauteur - 1], fill=0)

    else:
        raise ValueError("Forme inconnue : carre, rond, ovale ou none")

    return np.array(image)



def generer_nuage(texte: str, masque):
    wc = WordCloud(
        width=LARGEUR,
        height=HAUTEUR,
        mask=masque,
        max_words=NB_MOTS,
        background_color=FOND,
        colormap=COULEURS,
        collocations=False,
    )
    return wc.generate(texte)


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    texte = lire_texte(input_path)
    texte_filtre = filtrer_mots_courts(texte, LONGUEUR_MIN)

    # 1) Si après filtrage il ne reste rien, on s'arrête avec un message clair
    if not texte_filtre.strip():
        raise ValueError(
            "Après filtrage, il ne reste aucun mot à afficher.\n"
            f"Essaye de baisser LONGUEUR_MIN (actuel = {LONGUEUR_MIN}) "
            "ou vérifie que ton fichier contient bien du texte."
        )

    masque = creer_masque(FORME, LARGEUR, HAUTEUR)

    # 2) On tente avec le masque, et si ça échoue on réessaie sans masque
    try:
        nuage = generer_nuage(texte_filtre, masque)
    except ValueError as e:
        msg = str(e)
        if "Couldn't find space to draw" in msg:
            print("Attention : impossible de dessiner avec le masque. Nouvelle tentative sans masque.")
            nuage = generer_nuage(texte_filtre, None)
        else:
            raise

    # Sauvegarde
    nuage.to_image().save(output_path)

    # Affichage optionnel
    if PREVIEW:
        plt.imshow(nuage, interpolation="bilinear")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
