# Programme Python pour compter les mots "importants" d’un texte français
# - ignore les mots très courants (le, la, de, etc.)
# - regroupe les formes d’un même mot (verbe conjugué, pluriel, féminin…)
# - enlève les accents
# - produit un fichier CSV : mot canonique / nombre d’occurrences

# bibliothèques
import spacy
import csv
import unicodedata
from collections import Counter
from datetime import datetime

# variables
fichier_entree = "C:/PYTHON/.params/entree.txt"
fichier_sortie = "C:/PYTHON/.data/resultat.csv"

# Charger le modèle français
nlp = spacy.load("fr_core_news_sm")

def enlever_accents(texte):
        return "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )

def compter_mots_importants(fichier_texte, fichier_csv):
    # Lire le fichier texte
    print("{} - Lecture du fichier d entree".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(fichier_texte, "r", encoding="utf-8") as f:
        texte = f.read()

    # Analyse linguistique du texte
    print("{} - Analyse linguistique du texte".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    doc = nlp(texte)

    compteur = Counter()

    # Comptage des mots
    print("{} - Comptage des mots".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for token in doc:
        # Conditions pour garder seulement les mots "qui ont du sens"
        if (
            token.is_alpha              # uniquement des lettres
            and not token.is_stop       # pas de mots très courants (le, la, de…)
        ):
            # Forme canonique = lemme, minuscule, sans accent
            mot = token.lemma_.lower()
            mot = enlever_accents(mot)

            compteur[mot] += 1

    # Écriture du fichier CSV
    print("{} - Ecriture du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(fichier_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mot", "occurrences"])
        for mot, nb in compteur.most_common():
            writer.writerow([mot, nb])

def main():

    print("{} - Debut du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    compter_mots_importants(fichier_entree, fichier_sortie)
    print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if __name__ == "__main__":
    main()



