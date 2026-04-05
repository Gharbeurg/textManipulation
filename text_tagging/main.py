import sys
import os

# Permet les imports absolus depuis la racine du projet
sys.path.insert(0, os.path.dirname(__file__))

import config
from services import keyword_loader, text_normalizer, tagger, report_writer
from utils.console import info, success, error


def main() -> None:
    info("Chargement des mots-clés")
    try:
        keywords = keyword_loader.load(config.FICHIER_MOTS_CLES)
    except FileNotFoundError:
        error(f"Fichier mots-clés introuvable : {config.FICHIER_MOTS_CLES}")
        sys.exit(1)

    info("Lecture et tagging du fichier d'entrée")
    results = []

    try:
        with open(config.FICHIER_ENTREE, encoding=config.ENCODING, errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                phrase = text_normalizer.normalize(line)
                if not phrase:
                    continue
                result = tagger.tag(phrase, keywords)
                results.append(result)

                if result.tags:
                    success(f"{i} - {phrase}")
                else:
                    print(f"    {i} - {phrase}")

    except FileNotFoundError:
        error(f"Fichier d'entrée introuvable : {config.FICHIER_ENTREE}")
        sys.exit(1)
    except Exception as e:
        error(f"Erreur inattendue lors de la lecture : {e}")
        sys.exit(1)

    info("Création du fichier de sortie")
    try:
        report_writer.write(results, config.FICHIER_SORTIE)
    except Exception as e:
        error(f"Erreur lors de l'écriture du fichier de sortie : {e}")
        sys.exit(1)

    info("Fin du programme")


if __name__ == "__main__":
    main()
