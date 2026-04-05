from unidecode import unidecode


def normalize(text: str) -> str:
    return unidecode(text).lower().strip()
