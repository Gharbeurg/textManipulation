# bibliotheques
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from pathlib import Path

# variables
INPUT_FILE = Path("D:/CODING/.params/entree.txt")

def lire_fichier(chemin):
    """
    Lit un fichier texte et retourne son contenu
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    return chemin.read_text(encoding="utf-8")


def analyse_sentiment(texte):
    """
    Analyse le ton d'un texte et retourne les pourcentages
    positif / négatif / neutre
    """

    sia = SentimentIntensityAnalyzer()
    phrases = sent_tokenize(texte, language="french")

    if not phrases:
        return {
            "positif": 0.0,
            "negatif": 0.0,
            "neutre": 0.0
        }

    positif = negatif = neutre = 0

    for phrase in phrases:
        score = sia.polarity_scores(phrase)["compound"]

        if score > 0.05:
            positif += 1
        elif score < -0.05:
            negatif += 1
        else:
            neutre += 1

    total = len(phrases)

    return {
        "positif": round(100 * positif / total, 2),
        "negatif": round(100 * negatif / total, 2),
        "neutre": round(100 * neutre / total, 2)
    }


def main():
    texte = lire_fichier(INPUT_FILE)
    resultats = analyse_sentiment(texte)

    print("Analyse de sentiment (%):")
    print(f"Positif : {resultats['positif']} %")
    print(f"Négatif : {resultats['negatif']} %")
    print(f"Neutre  : {resultats['neutre']} %")


if __name__ == "__main__":
    main()
