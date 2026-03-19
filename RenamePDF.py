import os
from PyPDF2 import PdfReader

# dossier qui contient les PDF
PDF_DIR = r"C:/PYTHON/.entree/Sources"

def clean_filename(text):
    """
    enlève les caractères interdits dans les noms de fichiers Windows
    """
    forbidden = '<>:"/\\|?*'
    for c in forbidden:
        text = text.replace(c, "")
    return text.strip()

for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        old_path = os.path.join(PDF_DIR, filename)

        try:
            reader = PdfReader(old_path)

            # 1) on essaie de lire le titre dans les métadonnées du PDF
            title = reader.metadata.title if reader.metadata else None

            # 2) si pas de titre, on prend la première ligne de la première page
            if not title:
                first_page_text = reader.pages[0].extract_text()
                if first_page_text:
                    title = first_page_text.split("\n")[0]

            # 3) si toujours rien, on passe au fichier suivant
            if not title:
                print(f"Titre introuvable pour : {filename}")
                continue

            title = clean_filename(title)
            new_filename = title + ".pdf"
            new_path = os.path.join(PDF_DIR, new_filename)

            os.rename(old_path, new_path)
            print(f"Renommé : {filename} → {new_filename}")

        except Exception as e:
            print(f"Erreur avec {filename} : {e}")
