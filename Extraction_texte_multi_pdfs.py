import os
import fitz  # PyMuPDF

def extract_text_from_pdfs(directory, output_file):
    """
    Extrait le texte de tous les fichiers PDF dans un répertoire donné
    et l'enregistre dans un seul fichier texte.

    :param directory: Répertoire contenant les fichiers PDF.
    :param output_file: Nom du fichier texte de sortie.
    """
    with open(output_file, "w", encoding="utf-8") as out_file:
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                try:
                    doc = fitz.open(pdf_path)
                    text = "\n".join([page.get_text("text") for page in doc])
                    out_file.write(f"\n\n=== {filename} ===\n\n{text}\n")
                    print(f"Texte extrait de {filename}")
                except Exception as e:
                    print(f"Erreur avec {filename} : {e}")

if __name__ == "__main__":
    dossier_pdf = "C:/DATA/code/.data/pdfs"  # Remplace par le bon chemin
    fichier_sortie = "C:/DATA/code/.data/pdf_output.txt"
    extract_text_from_pdfs(dossier_pdf, fichier_sortie)
    print(f"Extraction terminée. Résultat dans {fichier_sortie}")
