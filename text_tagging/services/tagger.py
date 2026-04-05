from models.tag_result import TagResult


def tag(phrase: str, keywords: dict[str, list[str]]) -> TagResult:
    """
    Pour une phrase normalisée, retourne un TagResult avec les tags
    et mots-clés déclencheurs trouvés.
    Aucun I/O — pure fonction.
    """
    tags:     list[str] = []
    motscles: list[str] = []

    for tag_name, mots in keywords.items():
        for mot in mots:
            if mot and mot in phrase:
                tags.append(tag_name)
                motscles.append(mot)
                break  # un seul mot-clé déclencheur par tag suffit

    return TagResult(phrase=phrase, tags=tags, motscles=motscles)
