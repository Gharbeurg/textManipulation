import os
import re
from PyPDF2 import PdfReader

# dossier qui contient les PDF
PDF_DIR = r"C:/PYTHON/.entree/Sources"
BIBLIO_FILE = os.path.join(PDF_DIR, "bibliographie.txt")


def clean_filename(text):
    """
    Enlève les caractères interdits dans les noms de fichiers Windows
    """
    forbidden = '<>:"/\\|?*'
    for c in forbidden:
        text = text.replace(c, "")
    return text.strip()


def clean_text(text):
    """
    Nettoie un texte simple
    """
    if not text:
        return ""
    return " ".join(text.split()).strip()


def extract_year(reader, first_page_text=""):
    """
    Essaie de trouver une année dans :
    - les métadonnées
    - puis le texte de la première page

    On cherche une année plausible entre 1900 et 2099.
    """
    possible_texts = []

    if reader.metadata:
        meta = reader.metadata
        for field in [meta.title, meta.subject, meta.author, meta.creator, meta.producer]:
            if field:
                possible_texts.append(str(field))

        # date PDF typique : D:20240115123000
        if getattr(meta, "/CreationDate", None):
            possible_texts.append(str(meta["/CreationDate"]))
        if getattr(meta, "/ModDate", None):
            possible_texts.append(str(meta["/ModDate"]))

        # selon les versions, certains champs peuvent aussi être accessibles autrement
        for key in ["/CreationDate", "/ModDate"]:
            if key in meta:
                possible_texts.append(str(meta[key]))

    if first_page_text:
        possible_texts.append(first_page_text[:3000])  # limite raisonnable

    big_text = "\n".join(possible_texts)

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", big_text)
    for y in years:
        year = int(y)
        if 1900 <= year <= 2099:
            return str(year)

    return None


def extract_source_name(reader):
    """
    Essaie de trouver le nom de la source dans les métadonnées.
    Priorité :
    - author
    - creator
    - producer
    """
    if not reader.metadata:
        return None

    meta = reader.metadata

    candidates = [
        getattr(meta, "author", None),
        getattr(meta, "creator", None),
        getattr(meta, "producer", None),
    ]

    for value in candidates:
        value = clean_text(str(value)) if value else ""
        if value:
            return value

    # essai complémentaire avec accès par clé brute
    for key in ["/Author", "/Creator", "/Producer"]:
        if key in meta:
            value = clean_text(str(meta[key]))
            if value:
                return value

    return None


def write_bibliography_entry(file_handle, pdf_name, year=None, source=None):
    """
    Écrit une entrée dans bibliographie.txt.
    N'ajoute que les éléments trouvés.
    Si aucun élément n'est trouvé, n'écrit rien.
    """
    fields = []

    if pdf_name:
        fields.append(pdf_name)
    if year:
        fields.append(year)
    if source:
        fields.append(source)

    if fields:
        file_handle.write(" | ".join(fields) + "\n")


# 1) supprimer bibliographie.txt s'il existe
if os.path.exists(BIBLIO_FILE):
    os.remove(BIBLIO_FILE)

# 2) recréer le fichier vide
with open(BIBLIO_FILE, "w", encoding="utf-8") as bib_file:

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            old_path = os.path.join(PDF_DIR, filename)

            try:
                reader = PdfReader(old_path)

                # 1) on essaie de lire le titre dans les métadonnées du PDF
                title = reader.metadata.title if reader.metadata else None

                first_page_text = ""
                # 2) si pas de titre, on prend la première ligne de la première page
                if not title and len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text() or ""
                    if first_page_text:
                        title = first_page_text.split("\n")[0]

                # Si le titre existe déjà, on récupère quand même le texte de la 1re page
                # pour chercher l'année si ce n'est pas déjà fait
                if not first_page_text and len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text() or ""

                # 3) si toujours rien, on passe au fichier suivant
                if not title:
                    print(f"Titre introuvable pour : {filename}")
                    continue

                title = clean_filename(title)
                new_filename = title + ".pdf"
                new_path = os.path.join(PDF_DIR, new_filename)

                # éviter un problème si le nom est identique
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renommé : {filename} → {new_filename}")
                else:
                    print(f"Nom inchangé : {filename}")

                # extractions pour la bibliographie
                year = extract_year(reader, first_page_text)
                source = extract_source_name(reader)

                # 4) ajouter l'entrée dans bibliographie.txt
                write_bibliography_entry(
                    bib_file,
                    pdf_name=new_filename,
                    year=year,
                    source=source
                )

            except Exception as e:
                print(f"Erreur avec {filename} : {e}")
