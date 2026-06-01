import base64
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests

# supprime les avertissement de MuPDF, qui sont nombreux et pas pertinents pour notre usage 
try:
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
except Exception:
    pass


# ============================================================
# PARAMÈTRES PRINCIPAUX
# ============================================================

PDF_DIRECTORY = Path("C:/PYTHON/.entree/SourcesDocs")

# True  = simulation, aucun fichier renommé
# False = renommage réel
DRY_RUN = False

OLLAMA_URL = "http://localhost:11434/api/chat"

# Le modèle doit être capable de lire des images.
MODEL = "granite3.2-vision:2b"
# Exemples possibles :
# MODEL = "qwen2.5vl:7b"
# MODEL = "llama3.2-vision"

NB_PAGES_IMAGE = 2
NB_PAGES_TEXTE = 4

IMAGE_DPI = 160

MAX_TITLE_LENGTH = 100

IMAGES_OUTPUT_DIRECTORY = PDF_DIRECTORY / "_images_extraites"


# ============================================================
# STATISTIQUES
# ============================================================

def init_stats() -> dict:
    return {
        "pdf_trouves": 0,
        "pdf_traites": 0,
        "renommages_simules": 0,
        "renommages_reels": 0,
        "non_renommes_volontairement": 0,
        "erreurs": 0,
    }


def print_final_summary(stats: dict) -> None:
    print("=" * 80)
    print("RÉSUMÉ FINAL")
    print(f"PDF trouvés : {stats['pdf_trouves']}")
    print(f"PDF traités : {stats['pdf_traites']}")
    print(f"Renommages simulés : {stats['renommages_simules']}")
    print(f"Renommages réels : {stats['renommages_reels']}")
    print(f"Fichiers non renommés volontairement : {stats['non_renommes_volontairement']}")
    print(f"Erreurs : {stats['erreurs']}")
    print("=" * 80)


# ============================================================
# OUTILS TEXTE
# ============================================================

def remove_accents(text: str) -> str:
    """
    Supprime les accents.
    Exemple : ÉTUDE Santé -> ETUDE Sante
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def remove_file_extension_from_title(title: str) -> str:
    """
    Supprime les extensions parasites à la fin d'un titre.
    Exemple : Rapport final.pdf -> Rapport final
    """
    if not title:
        return ""

    return re.sub(
        r"\.(pdf|docx|doc|pptx|ppt|xlsx|xls)$",
        "",
        title.strip(),
        flags=re.IGNORECASE,
    )


def normalize_spaces(text: str) -> str:
    """
    Remplace les retours ligne et espaces multiples par un seul espace.
    """
    if not text:
        return ""

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# NETTOYAGE DES NOMS DE FICHIERS
# ============================================================

def clean_title_part(title: str, max_length: int = MAX_TITLE_LENGTH) -> str:
    """
    Nettoie uniquement le titre qui sera utilisé dans le nom final.

    Cette fonction :
    - supprime les extensions parasites ;
    - supprime les caractères interdits sous Windows ;
    - réduit les espaces ;
    - limite la longueur.
    """
    if not title:
        return ""

    title = remove_file_extension_from_title(title)
    title = normalize_spaces(title)

    # Caractères interdits sous Windows : \ / : * ? " < > |
    title = re.sub(r'[\\/:*?"<>|]', "_", title)

    # Supprime les points ou espaces finaux, problématiques sous Windows
    title = title.strip(" .")

    if max_length is not None:
        title = title[:max_length].strip(" .")

    return title


def clean_complete_filename(filename: str) -> str:
    """
    Nettoie un nom complet de fichier, extension comprise.

    Contrairement à clean_title_part(), cette fonction ne supprime pas .pdf.
    """
    if not filename:
        return "document.pdf"

    filename = normalize_spaces(filename)

    # Caractères interdits sous Windows : \ / : * ? " < > |
    filename = re.sub(r'[\\/:*?"<>|]', "_", filename)

    filename = filename.strip(" .")

    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"

    return filename


def clean_ollama_title(raw_title: str) -> str:
    """
    Nettoie la réponse d'Ollama pour ne garder que le titre.
    """
    if not raw_title:
        return "TITRE INCONNU"

    raw_title = raw_title.strip()

    lines = [line.strip() for line in raw_title.splitlines() if line.strip()]
    if not lines:
        return "TITRE INCONNU"

    # On garde la première ligne utile.
    title = lines[0]

    # Supprime les guillemets extérieurs
    title = title.strip('"').strip("'").strip("“”«»").strip()

    # Supprime les formules fréquentes ajoutées par le modèle
    parasite_patterns = [
        r"^le titre du document est\s*:\s*",
        r"^le titre est\s*:\s*",
        r"^voici le titre\s*:\s*",
        r"^titre du document\s*:\s*",
        r"^titre\s*:\s*",
        r"^document title\s*:\s*",
        r"^the title is\s*:\s*",
        r"^title\s*:\s*",
    ]

    for pattern in parasite_patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()

    title = title.strip('"').strip("'").strip("“”«»").strip()
    title = clean_title_part(title, max_length=MAX_TITLE_LENGTH)

    if not title:
        return "TITRE INCONNU"

    return title


def is_reliable_title(title: str) -> bool:
    """
    Règles retenues :
    - titre vide => non fiable
    - TITRE INCONNU => non fiable
    - commence par ERREUR => non fiable
    - moins de 5 caractères => non fiable
    """
    if not title:
        return False

    cleaned = title.strip()

    if not cleaned:
        return False

    if cleaned.upper() == "TITRE INCONNU":
        return False

    if cleaned.upper().startswith("ERREUR"):
        return False

    if len(cleaned) < 5:
        return False

    return True


# ============================================================
# GESTION DES CHEMINS DISPONIBLES
# ============================================================

def get_available_path(target_path: Path, original_path: Path | None = None) -> Path:
    """
    Retourne un chemin disponible.
    Si le fichier existe déjà, ajoute _2, _3, etc.

    original_path permet d'éviter de considérer le fichier courant
    comme un conflit avec lui-même.
    """
    if original_path is not None:
        try:
            if target_path.resolve() == original_path.resolve():
                return target_path
        except FileNotFoundError:
            pass

    if not target_path.exists():
        return target_path

    parent = target_path.parent
    stem = target_path.stem
    suffix = target_path.suffix

    counter = 2

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"

        if original_path is not None:
            try:
                if candidate.resolve() == original_path.resolve():
                    return candidate
            except FileNotFoundError:
                pass

        if not candidate.exists():
            return candidate

        counter += 1


# ============================================================
# ÉTAPE 1 — MISE EN MINUSCULES DU NOM INITIAL
# ============================================================

def build_lowercase_pdf_name(pdf_path: Path) -> str:
    """
    Met le nom initial en minuscules et supprime les accents.
    """
    stem = pdf_path.stem
    stem = remove_accents(stem)
    stem = stem.lower()
    stem = clean_title_part(stem, max_length=180)

    if not stem:
        stem = "document"

    return clean_complete_filename(f"{stem}.pdf")


def lowercase_pdf_filename(pdf_path: Path, stats: dict) -> Path:
    """
    Simule ou applique la mise en minuscules du nom initial.
    Retourne le chemin à utiliser pour la suite.
    """
    lowercase_name = build_lowercase_pdf_name(pdf_path)
    target_path = pdf_path.parent / lowercase_name
    available_path = get_available_path(target_path, original_path=pdf_path)

    print(f"ETAPE 1 - NOM MINUSCULE PROPOSÉ : {available_path.name}")

    # Aucun changement nécessaire
    if available_path.name == pdf_path.name:
        print("ETAPE 1 - ACTION : nom initial déjà conforme")
        return pdf_path

    if DRY_RUN:
        print("ETAPE 1 - ACTION : simulation, nom initial non modifié")
        return pdf_path

    try:
        pdf_path.rename(available_path)
        print("ETAPE 1 - ACTION : nom initial renommé en minuscules")
        return available_path

    except Exception as e:
        print(f"[ERREUR] Impossible de renommer en minuscules : {pdf_path.name} | {e}")
        print("ETAPE 1 - ACTION : poursuite avec le nom actuel")
        stats["erreurs"] += 1
        return pdf_path


# ============================================================
# ÉTAPE 2 — EXTRACTION DES IMAGES
# ============================================================

def ensure_images_output_directory() -> None:
    IMAGES_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def extract_first_pages_as_images(
    pdf_path: Path,
    output_dir: Path,
    max_pages: int,
) -> list[Path]:
    """
    Extrait les premières pages d'un PDF en images PNG.
    """
    image_paths = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERREUR] Impossible d'ouvrir le PDF : {pdf_path.name} | {e}")
        return image_paths

    try:
        pages_to_extract = min(max_pages, len(doc))
        safe_pdf_name = clean_title_part(remove_accents(pdf_path.stem.lower()), max_length=120)

        if not safe_pdf_name:
            safe_pdf_name = "document"

        for page_index in range(pages_to_extract):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=IMAGE_DPI)

            image_path = output_dir / f"{safe_pdf_name}_page_{page_index + 1}.png"

            # Évite un conflit éventuel de nom d'image
            image_path = get_available_path(image_path)

            pix.save(image_path)
            image_paths.append(image_path)

    except Exception as e:
        print(f"[ERREUR] Impossible d'extraire les images : {pdf_path.name} | {e}")

    finally:
        doc.close()

    return image_paths


def delete_images(image_paths: list[Path]) -> None:
    """
    Supprime les images créées pour un PDF.
    """
    for image_path in image_paths:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as e:
            print(f"[AVERTISSEMENT] Image non supprimée : {image_path.name} | {e}")


def delete_images_directory_if_empty() -> None:
    """
    Supprime le dossier _images_extraites s'il est vide.
    """
    try:
        if IMAGES_OUTPUT_DIRECTORY.exists() and IMAGES_OUTPUT_DIRECTORY.is_dir():
            if not any(IMAGES_OUTPUT_DIRECTORY.iterdir()):
                IMAGES_OUTPUT_DIRECTORY.rmdir()
    except Exception as e:
        print(f"[AVERTISSEMENT] Dossier images non supprimé : {e}")


# ============================================================
# ÉTAPE 3 — APPEL OLLAMA POUR TROUVER LE TITRE
# ============================================================

def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_ollama_for_title_single_image(image_path: Path) -> str:
    """
    Envoie une seule image à Ollama.
    """
    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        return f"ERREUR : impossible de lire l'image | {e}"

    prompt = "Quel est le titre sur cette image du document ? Réponds uniquement avec le titre."

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        return "ERREUR : Ollama ne semble pas lancé. Vérifie avec : ollama serve"

    except requests.exceptions.Timeout:
        return "ERREUR : délai dépassé pendant l'appel à Ollama"

    except requests.exceptions.HTTPError as e:
        return f"ERREUR HTTP Ollama : {e} | Réponse : {response.text}"

    except Exception as e:
        return f"ERREUR appel Ollama : {e}"

    try:
        data = response.json()
        message = data.get("message", {})
        raw_title = message.get("content", "").strip()

        if not raw_title:
            return "TITRE INCONNU"

        return clean_ollama_title(raw_title)

    except json.JSONDecodeError:
        return "ERREUR : réponse Ollama illisible"


def choose_best_title(candidates: list[str]) -> str:
    """
    Garde le premier titre fiable trouvé.
    """
    for candidate in candidates:
        title = clean_ollama_title(candidate)

        if is_reliable_title(title):
            return title

    return "TITRE INCONNU"


# ============================================================
# ÉTAPE 4 — EXTRACTION TEXTE ET RECHERCHE DE L'ANNÉE
# ============================================================

def extract_text_first_pages(pdf_path: Path, max_pages: int) -> str:
    """
    Extrait le texte des premières pages avec PyMuPDF.
    Pas d'OCR.
    """
    text_parts = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERREUR] Impossible d'ouvrir le PDF pour extraction texte : {pdf_path.name} | {e}")
        return ""

    try:
        pages_to_read = min(max_pages, len(doc))

        for page_index in range(pages_to_read):
            page = doc[page_index]
            text_parts.append(page.get_text("text"))

    except Exception as e:
        print(f"[ERREUR] Impossible d'extraire le texte : {pdf_path.name} | {e}")

    finally:
        doc.close()

    return "\n".join(text_parts)


def find_years_in_text(text: str) -> list[int]:
    """
    Trouve les années entre 1900 et 2099.
    """
    if not text:
        return []

    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)

    years = []

    for match in matches:
        try:
            year = int(match)
            if 1900 <= year <= 2099:
                years.append(year)
        except ValueError:
            continue

    return years


def extract_year_from_pdf_date_value(value: str) -> int | None:
    """
    Extrait une année depuis une valeur de métadonnée PDF.

    Exemples fréquents :
    - D:20240115103000
    - 2024-01-15
    - 2024
    """
    if not value:
        return None

    match = re.search(r"\b(19\d{2}|20\d{2})\b", value)

    if not match:
        # Cas fréquent des dates PDF : D:20240115103000
        match = re.search(r"D:(19\d{2}|20\d{2})", value)

    if not match:
        return None

    try:
        year = int(match.group(1))
    except ValueError:
        return None

    if 1900 <= year <= 2099:
        return year

    return None


def find_year_in_pdf_metadata(pdf_path: Path) -> int | None:
    """
    Cherche une année dans les métadonnées internes du PDF.

    Priorité :
    1. date de création PDF ;
    2. date de modification PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible de lire les métadonnées PDF : {pdf_path.name} | {e}")
        return None

    try:
        metadata = doc.metadata or {}

        for key in ("creationDate", "modDate"):
            year = extract_year_from_pdf_date_value(metadata.get(key, ""))

            if year is not None:
                return year

    except Exception as e:
        print(f"[AVERTISSEMENT] Métadonnées PDF illisibles : {pdf_path.name} | {e}")

    finally:
        doc.close()

    return None


def find_year_in_file_system_metadata(pdf_path: Path) -> int | None:
    """
    Cherche une année dans les métadonnées système du fichier.

    Sous Windows, st_ctime correspond en général à la date de création.
    Sous Linux/macOS, st_ctime correspond plutôt à la date de changement
    des métadonnées. On garde donc st_mtime en second choix.
    """
    try:
        file_stat = pdf_path.stat()
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible de lire les métadonnées système : {pdf_path.name} | {e}")
        return None

    timestamps = [file_stat.st_ctime, file_stat.st_mtime]

    for timestamp in timestamps:
        try:
            year = datetime.fromtimestamp(timestamp).year
        except Exception:
            continue

        if 1900 <= year <= 2099:
            return year

    return None


def find_year_in_metadata(pdf_path: Path) -> str:
    """
    Retourne l'année trouvée dans les métadonnées du PDF ou du fichier.
    """
    pdf_metadata_year = find_year_in_pdf_metadata(pdf_path)

    if pdf_metadata_year is not None:
        return str(pdf_metadata_year)

    file_system_year = find_year_in_file_system_metadata(pdf_path)

    if file_system_year is not None:
        return str(file_system_year)

    return "ANNEE INCONNUE"


def choose_document_year(pdf_path: Path) -> tuple[str, str]:
    """
    Choisit l'année à utiliser pour le nom final.

    Règle :
    - on cherche d'abord l'année dans les métadonnées du PDF ou du fichier ;
    - si aucune année exploitable n'est trouvée dans les métadonnées,
      on cherche l'année dans le texte des premières pages ;
    - si aucune année n'est trouvée dans les deux sources,
      on retourne ANNEE INCONNUE.

    Retourne :
    - l'année ;
    - la source utilisée, pour affichage dans la console.
    """
    metadata_year = find_year_in_metadata(pdf_path)

    if metadata_year != "ANNEE INCONNUE":
        return metadata_year, "métadonnées fichier"

    text = extract_text_first_pages(
        pdf_path=pdf_path,
        max_pages=NB_PAGES_TEXTE,
    )

    years = find_years_in_text(text)

    if years:
        current_year = datetime.now().year
        valid_years = [year for year in years if year <= current_year]

        if valid_years:
            return str(max(valid_years)), "texte, car année absente des métadonnées"

        future_year = max(years)
        return "ANNEE INCONNUE", f"aucune année valide : métadonnées absentes et année texte future : {future_year}"

    return "ANNEE INCONNUE", "aucune année trouvée dans les métadonnées ni dans le texte"


# ============================================================
# ÉTAPE 5 — CONSTRUCTION DU NOM FINAL
# ============================================================

def build_final_pdf_name(year: str, title: str) -> str:
    """
    Construit le nom final :
    ANNEE - TITRE.pdf
    """
    year_part = remove_accents(year).upper().strip()
    title_part = remove_accents(title).upper().strip()

    title_part = clean_title_part(title_part, max_length=MAX_TITLE_LENGTH)

    if not title_part:
        title_part = "TITRE INCONNU"

    final_name = f"{year_part} - {title_part}.pdf"
    final_name = clean_complete_filename(final_name)

    return final_name


# ============================================================
# ÉTAPE 6 — RENOMMAGE FINAL
# ============================================================

def rename_pdf_final(
    current_pdf_path: Path,
    final_pdf_path: Path,
    stats: dict,
) -> bool:
    """
    Simule ou applique le renommage final.
    """
    print(f"ANCIEN NOM : {current_pdf_path.name}")
    print(f"NOUVEAU NOM PROPOSÉ : {final_pdf_path.name}")

    if DRY_RUN:
        print("ACTION : simulation, fichier non renommé")
        stats["renommages_simules"] += 1
        return True

    try:
        if final_pdf_path.name == current_pdf_path.name:
            print("ACTION : fichier déjà au bon nom")
            stats["non_renommes_volontairement"] += 1
            return True

        current_pdf_path.rename(final_pdf_path)
        print("ACTION : fichier renommé")
        stats["renommages_reels"] += 1
        return True

    except Exception as e:
        print(f"[ERREUR] Impossible de renommer le fichier final : {current_pdf_path.name} | {e}")
        stats["erreurs"] += 1
        return False


# ============================================================
# TRAITEMENT D'UN PDF
# ============================================================

def process_pdf(pdf_path: Path, stats: dict) -> None:
    """
    Traitement complet d'un PDF.
    """
    stats["pdf_traites"] += 1

    print("=" * 80)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - traitement : {pdf_path.name}")
    print()

    current_pdf_path = pdf_path
    image_paths = []

    try:
        # Étape 1 : mise en minuscules / suppression accents du nom initial
        current_pdf_path = lowercase_pdf_filename(current_pdf_path, stats)

        # Étape 2 : extraction des images
        ensure_images_output_directory()

        image_paths = extract_first_pages_as_images(
            pdf_path=current_pdf_path,
            output_dir=IMAGES_OUTPUT_DIRECTORY,
            max_pages=NB_PAGES_IMAGE,
        )

        if not image_paths:
            print("TITRE TROUVÉ : TITRE INCONNU")
            print("NOUVEAU NOM PROPOSÉ : aucun")
            print("ACTION : fichier non renommé, images non générées")
            stats["non_renommes_volontairement"] += 1
            return

        # Étape 3 : recherche du titre avec Ollama
        candidates = []

        for index, image_path in enumerate(image_paths, start=1):
            title = ask_ollama_for_title_single_image(image_path)
            candidates.append(title)
            print(f"PAGE {index} - TITRE CANDIDAT : {title}")

        best_title = choose_best_title(candidates)

        print(f"ETAPE 2 - TITRE TROUVÉ : {best_title}")

        if not is_reliable_title(best_title):
            print()
            print("NOUVEAU NOM PROPOSÉ : aucun")
            print("ACTION : fichier non renommé, titre non fiable")
            stats["non_renommes_volontairement"] += 1
            return

        # Étape 4 : recherche année
        # Priorité aux métadonnées du PDF ou du fichier.
        # Le texte des premières pages n'est utilisé qu'en secours.
        document_year, document_year_source = choose_document_year(current_pdf_path)

        print(f"ETAPE 3 - ANNEE TROUVÉE : {document_year}")
        print(f"ETAPE 3 - SOURCE ANNEE : {document_year_source}")
        print()

        # Étape 5 : construction du nom final
        final_name = build_final_pdf_name(document_year, best_title)
        target_final_path = current_pdf_path.parent / final_name

        available_final_path = get_available_path(
            target_final_path,
            original_path=current_pdf_path,
        )

        # Étape 6 : renommage final
        rename_pdf_final(
            current_pdf_path=current_pdf_path,
            final_pdf_path=available_final_path,
            stats=stats,
        )

    except Exception as e:
        print(f"[ERREUR] Erreur inattendue sur le fichier : {pdf_path.name} | {e}")
        stats["erreurs"] += 1

    finally:
        # Sécurité importante : les images sont supprimées même si une erreur survient.
        delete_images(image_paths)


# ============================================================
# PARCOURS DU RÉPERTOIRE
# ============================================================

def process_directory(directory_path: Path, stats: dict) -> None:
    """
    Parcourt le répertoire principal et traite tous les PDF.
    Ne traite pas les sous-répertoires.
    """
    if not directory_path.exists():
        print(f"[ERREUR] Le répertoire n'existe pas : {directory_path}")
        stats["erreurs"] += 1
        return

    if not directory_path.is_dir():
        print(f"[ERREUR] Le chemin n'est pas un répertoire : {directory_path}")
        stats["erreurs"] += 1
        return

    pdf_files = sorted(directory_path.glob("*.pdf"))

    stats["pdf_trouves"] = len(pdf_files)

    if not pdf_files:
        print(f"Aucun fichier PDF trouvé dans : {directory_path}")
        return

    print("=" * 80)
    print(f"Répertoire PDF : {directory_path}")
    print(f"Mode DRY_RUN : {DRY_RUN}")
    print(f"Modèle Ollama : {MODEL}")
    print(f"Nombre de fichiers PDF trouvés : {len(pdf_files)}")
    print("=" * 80)

    for pdf_path in pdf_files:
        process_pdf(pdf_path, stats)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main() -> None:
    stats = init_stats()

    process_directory(PDF_DIRECTORY, stats)

    delete_images_directory_if_empty()

    print_final_summary(stats)


if __name__ == "__main__":
    main()