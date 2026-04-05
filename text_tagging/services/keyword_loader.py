import re
import config


def load(filepath: str) -> dict[str, list[str]]:
    """
    Lit motscles.txt et retourne un dict {tag: [motcle1, motcle2, ...]}.
    Format attendu : #TAG,motcle1,motcle2,...
    Gère :
      - le séparateur ';' à la place de ',' entre le tag et les mots-clés (typo connue)
      - les tags dupliqués (fusion des listes de mots-clés)
      - les lignes vides
    """
    keywords: dict[str, list[str]] = {}

    with open(filepath, encoding=config.ENCODING, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Accepte ',' ou ';' comme premier séparateur (ex: #KINE;kinesytherap,kine)
            first_sep = re.search(r"[,;]", line)
            if not first_sep:
                continue

            tag  = line[:first_sep.start()].strip()
            rest = line[first_sep.start() + 1:]

            if not tag:
                continue

            mots = [m.strip() for m in rest.split(config.SEPARATEUR_CSV) if m.strip()]
            if not mots:
                continue

            # Fusion si le tag existe déjà (ex: #SYMPT-DYSPNEE sur deux lignes)
            if tag in keywords:
                existing = set(keywords[tag])
                keywords[tag].extend(m for m in mots if m not in existing)
            else:
                keywords[tag] = mots

    return keywords
