# -*- coding: utf-8 -*-
"""
Convertisseur multi-format -> Markdown avec Docling
Formats supportés : PDF, DOCX, PPTX, XLSX, HTML, images, AsciiDoc, Markdown, CSV
Usage : python Extraction_markdown_from_docling_corrige.py
Installation : pip install docling

Objectif de cette version :
- forcer Docling à utiliser un répertoire local commun pour ses modèles ;
- forcer RapidOCR à utiliser les fichiers de modèles placés dans ce même répertoire ;
- éviter que RapidOCR retourne chercher ses modèles dans le dossier site-packages du venv.
"""

from pathlib import Path
from datetime import datetime
import logging
import os


# -----------------------------------------------------------------------------
# PARAMÈTRES PRINCIPAUX
# -----------------------------------------------------------------------------

INPUT_DIR = Path(r"C:/PYTHON/.entree/SourcesDocs")
OUTPUT_DIR = Path(r"C:/PYTHON/.data/ResultatsMarkdown")

# Répertoire unique où l'on veut stocker et réutiliser les modèles Docling/RapidOCR.
MODELS_DIR = Path(r"C:/PYTHON/.params/.models-doclings")

# Noms des modèles RapidOCR vus dans tes logs.
# Les fichiers doivent être copiés dans MODELS_DIR.
RAPIDOCR_DET_MODEL = MODELS_DIR / "ch_PP-OCRv4_det_infer.pth"
RAPIDOCR_CLS_MODEL = MODELS_DIR / "ch_ptocr_mobile_v2.0_cls_infer.pth"
RAPIDOCR_REC_MODEL = MODELS_DIR / "ch_PP-OCRv4_rec_infer.pth"

# Très important : cette variable doit être définie AVANT d'importer Docling.
# Elle indique à Docling où chercher ses modèles et artefacts locaux.
os.environ["DOCLING_ARTIFACTS_PATH"] = str(MODELS_DIR)


from docling.document_converter import DocumentConverter  # noqa: E402

try:
    # Imports utilisés par les versions récentes de Docling.
    from docling.document_converter import PdfFormatOption, InputFormat  # noqa: E402
except ImportError:  # pragma: no cover
    # Compatibilité avec certaines versions plus anciennes.
    from docling.document_converter import PdfFormatOption  # noqa: E402
    from docling.datamodel.base_models import InputFormat  # noqa: E402

from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions  # noqa: E402


# Extensions reconnues par Docling et le suffixe InputFormat correspondant
SUPPORTED_EXTENSIONS = {
    # Documents Office
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    # Images (Docling applique OCR + layout)
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp",
    # Web / texte structuré
    ".html", ".htm", ".mhtml",
    ".md",
    ".adoc", ".asciidoc",
    ".csv",
}


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")


def clear_output_dir(directory: Path) -> None:
    """Supprime tous les fichiers du répertoire de sortie, sans supprimer le répertoire."""
    if not directory.exists():
        log("Répertoire de sortie inexistant, il sera créé à la première conversion.")
        return

    files = [f for f in directory.iterdir() if f.is_file()]
    if not files:
        log("Répertoire de sortie déjà vide.")
        return

    for f in files:
        f.unlink()
    log(f"Répertoire de sortie vidé ({len(files)} fichier(s) supprimé(s)).")


def collect_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """
    Parcourt le répertoire non récursivement et sépare :
    - les fichiers à convertir ;
    - les fichiers ignorés.
    """
    supported, ignored = [], []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(f)
        else:
            ignored.append(f)
    return supported, ignored


def check_models_dir() -> None:
    """Vérifie le dossier de modèles et affiche les fichiers RapidOCR manquants."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Répertoire modèles demandé : {MODELS_DIR}")
    log(f"Variable DOCLING_ARTIFACTS_PATH : {os.environ.get('DOCLING_ARTIFACTS_PATH')}")

    expected_models = [
        RAPIDOCR_DET_MODEL,
        RAPIDOCR_CLS_MODEL,
        RAPIDOCR_REC_MODEL,
    ]
    missing = [p for p in expected_models if not p.exists()]

    if missing:
        log("ATTENTION : modèle(s) RapidOCR absent(s) du répertoire demandé :")
        for p in missing:
            log(f"  - {p}")
        log("RapidOCR peut alors revenir à son comportement par défaut ou échouer selon ta version de Docling.")
        log("Copie les fichiers .pth depuis le dossier rapidocr/models de ton environnement virtuel vers ce répertoire.")
    else:
        log("Modèles RapidOCR trouvés dans le répertoire demandé.")


def build_converter() -> DocumentConverter:
    """
    Crée un convertisseur Docling configuré pour les PDF.

    Le point clé est RapidOcrOptions : on donne explicitement les 3 chemins de modèles.
    Comme cela, RapidOCR ne doit plus aller chercher ses fichiers dans site-packages/rapidocr/models.
    """
    ocr_options = RapidOcrOptions(
        # IMPORTANT : tes fichiers sont des modèles PyTorch .pth.
        # Sans ce réglage, Docling/RapidOCR peut choisir le backend par défaut
        # "onnxruntime", qui attend onnxruntime + des modèles .onnx.
        backend="torch",
        det_model_path=str(RAPIDOCR_DET_MODEL),
        cls_model_path=str(RAPIDOCR_CLS_MODEL),
        rec_model_path=str(RAPIDOCR_REC_MODEL),
    )

    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(MODELS_DIR),
        ocr_options=ocr_options,
    )
    pipeline_options.do_ocr = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def convert_file(path: Path, out_dir: Path, converter: DocumentConverter) -> bool:
    """Convertit un fichier en Markdown. Retourne True si succès."""
    try:
        log(f"Conversion [{path.suffix.upper()}] : {path.name}")
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.md"
        out_path.write_text(markdown, encoding="utf-8")

        log(f"  -> Sauvegardé : {out_path.name}")
        return True

    except Exception as e:
        log(f"  !! ERREUR ({path.name}) : {type(e).__name__} : {e}")
        return False


def main() -> None:
    logging.getLogger("docling").setLevel(logging.ERROR)

    log("START")
    check_models_dir()
    clear_output_dir(OUTPUT_DIR)

    if not INPUT_DIR.exists():
        log(f"Répertoire source introuvable : {INPUT_DIR}")
        return

    supported, ignored = collect_files(INPUT_DIR)

    if ignored:
        log(f"{len(ignored)} fichier(s) ignoré(s) (format non supporté) :")
        for f in ignored:
            log(f"  - {f.name}")

    if not supported:
        log("Aucun fichier convertible trouvé.")
        return

    log(f"{len(supported)} fichier(s) à convertir.")

    converter = build_converter()
    success = 0

    for path in supported:
        if convert_file(path, OUTPUT_DIR, converter):
            success += 1

    log(f"DONE - {success}/{len(supported)} fichier(s) converti(s) avec succès.")


if __name__ == "__main__":
    main()
