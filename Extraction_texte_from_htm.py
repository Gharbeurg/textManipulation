from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional, List

from bs4 import BeautifulSoup, Tag
import trafilatura
import xml.etree.ElementTree as ET


# =========================
# PARAMETRES A MODIFIER
# =========================
SOURCE_DIR = Path(r"C:/PYTHON/.entree/SourcesHTM")
OUTPUT_DIR = Path(r"C:/PYTHON/.data/ResultatsHTM")

# Longueur minimale d'une ligne pour être conservée
MIN_LINE_LENGTH = 15

# Si True, garde les crédits image du type © ...
KEEP_IMAGE_CREDITS = False

# Si True, garde les blocs "pour aller plus loin", "actualités", etc.
KEEP_RELATED_CONTENT = False


# =========================
# BRUIT STRUCTUREL : SELECTEURS CSS
# =========================
NOISE_SELECTORS = [
    # Scripts / styles / objets techniques
    "script", "style", "noscript", "iframe", "svg", "canvas", "form",

    # Navigation générale
    "header", "footer", "nav", "aside",
    "[role='navigation']",
    "[aria-label*='breadcrumb' i]",
    "ol.breadcrumb",
    "ul.breadcrumb",
    ".breadcrumb",

    # Menus / recherche / partage / widgets
    ".menu", ".navbar", ".sidebar", ".search", ".searchform",
    ".share", ".social", ".newsletter", ".cookie", ".banner", ".popup",
    ".advert", ".ads", ".ad", ".promo",

    # Widgets interactifs fréquents
    "div.ameli-quiz-js",
    "div.ameli-quiz",
    "[class*='quiz']",
    "div.share-url-wrapper",
    "ul.links.list-inline",
    "div.text-right",
    "button.accordion-control",
    "div.wrapper-transcription-textuelle",

    # Blocs souvent annexes
    ".related", ".related-content", ".recommended", ".recommendation",
    ".read-more", ".discover", ".cta", ".promo-block",
    ".latest-news", ".news-list", ".sharing-tools",
]


# =========================
# BRUIT STRUCTUREL : PATTERNS DE CLASSE / ID
# =========================
NOISE_CLASS_PATTERN = re.compile(
    r"(menu|nav|navbar|sidebar|footer|header|breadcrumb|social|share|"
    r"cookie|banner|popup|comment|newsletter|related|recommend|promo|"
    r"advert|ads?\b|cta|search|toolbar|subscription|follow|faq|"
    r"pagination|pager|metadata-extra|utility|widget)",
    re.I
)

RELATED_SECTION_PATTERN = re.compile(
    r"(pour aller plus loin|à découvrir aussi|actualités|articles liés|"
    r"contenus liés|sur le même sujet|en savoir plus|voir aussi|"
    r"related|read more|discover more|latest news)",
    re.I
)

NON_INFORMATIVE_LINE_PATTERN = re.compile(
    r"^(tout afficher|tout replier|en savoir plus|lire aussi|partager|"
    r"imprimer|retour en haut|consulter|cliquez ici|choisir la langue|"
    r"page d'accueil|accès direct au contenu|accès direct au menu principal|"
    r"recherche|chercher)$",
    re.I
)

DATE_ONLY_PATTERN = re.compile(
    r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\d{4})$",
    re.I
)


# =========================
# OUTILS TEXTE
# =========================
def looks_like_heading_candidate(text: str) -> bool:
    """Détecte un paragraphe court qui ressemble à un intertitre."""
    if not text:
        return False

    t = normalize_extracted_text(text)

    if len(t) < 8 or len(t) > 80:
        return False

    if is_suspect_noise_line(t):
        return False

    lowered = t.lower()

    # Evite les phrases complètes trop narratives
    if t.endswith(".") or t.endswith("!") or t.endswith("?") or t.endswith(":"):
        return False

    # Evite les lignes trop longues en mots
    words = t.split()
    if len(words) > 8:
        return False

    # Evite les lignes purement numériques
    if re.fullmatch(r"[\d\s%/.,:;()\-–—]+", t):
        return False

    return True

def clean_text(text: Optional[str]) -> str:
    """Nettoyage léger."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_extracted_text(text: Optional[str]) -> str:
    """Nettoyage renforcé du texte extrait."""
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("’ ", "’")
    text = text.replace(" '", "'")
    text = text.replace("“ ", "“").replace(" ”", "”")

    # Réduction des espaces
    text = re.sub(r"[ \t]+", " ", text)

    # Supprime les espaces avant ponctuation
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)

    # Supprime les espaces après parenthèse ouvrante
    text = re.sub(r"([(\[])\s+", r"\1", text)

    # Corrige les espaces autour des apostrophes
    text = re.sub(r"\b([ldjtmcsnLDJTMCSN])\s*'\s*", r"\1'", text)

    # Corrige les espaces avant/après slash
    text = re.sub(r"\s*/\s*", "/", text)

    # Corrige les espaces avant % si besoin
    text = re.sub(r"(\d)\s+%", r"\1%", text)

    # Corrige les points de suspension
    text = re.sub(r"\.\.\.+", "...", text)

    # Corrige quelques artefacts fréquents simples
    text = re.sub(r"\blacause\b", "la cause", text, flags=re.I)
    text = re.sub(r"\bdécroit\b", "décroît", text, flags=re.I)
    text = re.sub(r"\beux personnes\b", "les personnes", text, flags=re.I)

    # Nettoie espaces multiples après corrections
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_informative_line(text: str, tag_label: str = "") -> bool:
    """Filtre les lignes peu informatives."""
    if not text:
        return False

    t = normalize_extracted_text(text)

    if len(t) < MIN_LINE_LENGTH:
        return False

    if NON_INFORMATIVE_LINE_PATTERN.match(t):
        return False

    if DATE_ONLY_PATTERN.match(t):
        return False

    if not KEEP_IMAGE_CREDITS and t.startswith("©"):
        return False

    # Lignes presque uniquement numériques / symboliques
    if re.fullmatch(r"[\d\s%/.,:;()\-–—]+", t):
        return False

    # Évite des lignes trop "interface"
    lowered = t.lower()
    interface_words = [
        "choisir la langue", "page d'accueil", "retour à l’accueil",
        "lancer la recherche", "fermer", "menu", "partager", "imprimer"
    ]
    if any(w in lowered for w in interface_words):
        return False

    return True


def clear_output_directory(output_dir: Path) -> None:
    """Supprime tous les fichiers du répertoire de sortie."""
    if not output_dir.exists():
        return
    for item in output_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"Impossible de supprimer {item} : {e}")


def safe_filename(name: str) -> str:
    """Produit un nom de fichier sûr."""
    name = Path(name).stem
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name or "sortie"


# =========================
# HEURISTIQUES CONTENEUR PRINCIPAL
# =========================
def score_container(tag: Tag) -> int:
    """
    Donne un score simple à un conteneur :
    + plus il contient de paragraphes/listes/titres, plus il est intéressant
    - on pénalise les conteneurs très orientés navigation
    """
    if not isinstance(tag, Tag):
        return -10_000

    text = normalize_extracted_text(tag.get_text(" ", strip=True))
    if len(text) < 200:
        return -10_000

    paragraphs = len(tag.find_all("p"))
    items = len(tag.find_all("li"))
    headings = len(tag.find_all(["h1", "h2", "h3", "h4"]))
    figcaptions = len(tag.find_all("figcaption"))

    classes = " ".join(tag.get("class", []))
    tag_id = tag.get("id", "")
    attrs_blob = f"{tag.name} {classes} {tag_id}".lower()

    penalty = 0
    if NOISE_CLASS_PATTERN.search(attrs_blob):
        penalty += 100

    bonus = 0
    if any(k in attrs_blob for k in [
        "article", "content", "main", "entry", "post", "body", "wrapper"
    ]):
        bonus += 40

    score = (
        len(text) // 100
        + paragraphs * 12
        + items * 4
        + headings * 10
        + figcaptions * 3
        + bonus
        - penalty
    )
    return score


def find_best_main_container(soup: BeautifulSoup) -> Optional[Tag]:
    """Cherche le meilleur conteneur principal."""
    candidates: List[Tag] = []

    # Priorité aux zones sémantiques usuelles
    priority_selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article__wrapper",
        ".content",
        ".main-content",
        ".entry-content",
        ".post-content",
        ".article-content",
        ".wysiwyg",
    ]

    for selector in priority_selectors:
        for tag in soup.select(selector):
            if isinstance(tag, Tag):
                candidates.append(tag)

    # Si on a peu de candidats, on ajoute des div/section potentiellement utiles
    if len(candidates) < 3:
        for tag in soup.find_all(["div", "section"]):
            if not isinstance(tag, Tag):
                continue
            txt = normalize_extracted_text(tag.get_text(" ", strip=True))
            if len(txt) >= 300:
                candidates.append(tag)

    if not candidates:
        return None

    best = max(candidates, key=score_container)
    if score_container(best) < 0:
        return None
    return best


# =========================
# PRE-TRAITEMENT HTML
# =========================
def preprocess_html(html: str) -> str:
    """
    Nettoie le HTML avant extraction.
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. Remplace les boutons tooltips / glossaires par leur texte
    for btn in soup.find_all("button", class_=re.compile(r"onomasticon", re.I)):
        btn.replace_with(btn.get_text(" ", strip=True))

    # 2. Déplie les titres accordéons
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        btns = heading.find_all("button")
        if btns:
            title_text = " ".join(
                b.get_text(" ", strip=True) for b in btns if b.get_text(strip=True)
            )
            if title_text:
                for b in btns:
                    b.decompose()
                heading.clear()
                heading.append(title_text)

    # 3. Supprime les sélecteurs de bruit
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # 4. Supprime selon classes / ids bruités
    for tag in list(soup.find_all(True)):
        if not isinstance(tag, Tag):
            continue

        # Tag déjà supprimé ou devenu invalide
        if getattr(tag, "attrs", None) is None:
            continue

        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        blob = f"{classes} {tag_id}"

        if NOISE_CLASS_PATTERN.search(blob):
            if tag.name not in ("article", "main"):
                tag.decompose()

    # 5. Option : retire les sections "contenus liés"
    if not KEEP_RELATED_CONTENT:
        for heading in soup.find_all(["h2", "h3", "h4"]):
            heading_text = normalize_extracted_text(heading.get_text(" ", strip=True))
            if RELATED_SECTION_PATTERN.search(heading_text):
                parent = heading.parent if isinstance(heading.parent, Tag) else None
                # Si la section est dans un bloc identifiable, on retire ce bloc
                if parent and parent.name in ("section", "div"):
                    parent.decompose()
                else:
                    heading.decompose()

    return str(soup)


# =========================
# EXTRACTION METADONNEES
# =========================
def extract_meta_from_html(html: str) -> dict:
    """Extrait quelques métadonnées utiles."""
    soup = BeautifulSoup(html, "lxml")

    result = {"title": "", "summary": ""}

    if soup.title and soup.title.string:
        result["title"] = normalize_extracted_text(soup.title.string)

    meta_description = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta_description and meta_description.get("content"):
        result["summary"] = normalize_extracted_text(meta_description.get("content"))

    if not result["summary"]:
        og_desc = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
        if og_desc and og_desc.get("content"):
            result["summary"] = normalize_extracted_text(og_desc.get("content"))

    return result


# =========================
# EXTRACTION PRINCIPALE (trafilatura)
# =========================
def trafilatura_extract_xml(html: str, favor_precision: bool = True) -> Optional[str]:
    """Extrait le contenu principal sous forme XML structurée."""
    try:
        extracted = trafilatura.extract(
            html,
            output_format="xml",
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_formatting=True,
            include_links=False,
            favor_precision=favor_precision,
            deduplicate=True,
        )
        return extracted
    except Exception:
        return None


def map_heading_level_from_text(text: str) -> str:
    """
    Repli si trafilatura ne donne pas le niveau du titre.
    Par défaut, on garde un niveau générique.
    """
    return "[INTERTITRE]"


def parse_trafilatura_xml(xml_text: str) -> List[str]:
    """
    Transforme le XML extrait en texte structuré.
    """
    lines: List[str] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return lines

    for elem in root.iter():
        tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
        text = normalize_extracted_text("".join(elem.itertext()))

        if not text:
            continue

        if tag == "head":
            if is_informative_line(text):
                # Trafilatura ne fournit pas toujours le niveau exact de titre
                lines.append(map_heading_level_from_text(text) + f" {text}")

        elif tag == "p":
            if is_informative_line(text, "p"):
                lines.append(f"[PARAGRAPHE] {text}")

        elif tag == "item":
            if is_informative_line(text, "li"):
                lines.append(f"[LISTE] {text}")

        elif tag == "quote":
            if is_informative_line(text, "quote"):
                lines.append(f"[CITATION] {text}")

        elif tag == "figdesc":
            if is_informative_line(text, "figcaption"):
                lines.append(f"[LEGENDE] {text}")

        elif tag == "cell":
            if is_informative_line(text, "cell"):
                lines.append(f"[TABLEAU] {text}")

    return deduplicate_preserve_order(lines)


# =========================
# EXTRACTION DE SECOURS (BS4)
# =========================
def bs4_heading_label(tag_name: str) -> str:
    if tag_name == "h1":
        return "[TITRE_ARTICLE]"
    if tag_name == "h2":
        return "[H2]"
    if tag_name == "h3":
        return "[H3]"
    if tag_name == "h4":
        return "[H4]"
    return "[INTERTITRE]"


def _bs4_extract_lines(container: Tag) -> List[str]:
    """Extrait les lignes structurées depuis un conteneur BS4."""
    lines: List[str] = []

    for tag in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "figcaption"]):
        text = normalize_extracted_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        if tag.name in ("h1", "h2", "h3", "h4"):
            if is_informative_line(text, tag.name):
                lines.append(f"{bs4_heading_label(tag.name)} {text}")

        elif tag.name == "p":
            if is_informative_line(text, "p"):
                lines.append(f"[PARAGRAPHE] {text}")

        elif tag.name == "li":
            if is_informative_line(text, "li"):
                lines.append(f"[LISTE] {text}")

        elif tag.name == "figcaption":
            if is_informative_line(text, "figcaption"):
                lines.append(f"[LEGENDE] {text}")

    return lines


def remove_noise_inside_container(container: Tag) -> None:
    """Supprime le bruit dans un conteneur."""
    for selector in NOISE_SELECTORS:
        for tag in container.select(selector):
            if isinstance(tag, Tag):
                tag.decompose()

    for tag in list(container.find_all(True)):
        if not isinstance(tag, Tag):
            continue

        # Balise devenue invalide après suppression d'un parent
        if getattr(tag, "attrs", None) is None:
            continue

        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        blob = f"{classes} {tag_id}"

        if NOISE_CLASS_PATTERN.search(blob):
            if tag.name not in ("article", "main"):
                tag.decompose()


def fallback_extract_with_bs4(html: str) -> List[str]:
    """
    Fallback :
    1. chercher le meilleur conteneur principal
    2. sinon article/main
    3. sinon page entière nettoyée
    """
    soup = BeautifulSoup(html, "lxml")

    # Tentative 1 : meilleur conteneur principal
    best_container = find_best_main_container(soup)
    if isinstance(best_container, Tag):
        remove_noise_inside_container(best_container)
        lines = _bs4_extract_lines(best_container)
        if len(lines) >= 3:
            return deduplicate_preserve_order(lines)

    # Tentative 2 : article/main
    zone = soup.find("article") or soup.find("main")
    if isinstance(zone, Tag):
        remove_noise_inside_container(zone)
        lines = _bs4_extract_lines(zone)
        if len(lines) >= 3:
            return deduplicate_preserve_order(lines)

    # Tentative 3 : document complet nettoyé
    for tag in list(soup.find_all(True)):
        if not isinstance(tag, Tag):
            continue

        # Balise devenue invalide après suppression d'un parent
        if getattr(tag, "attrs", None) is None:
            continue

        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        blob = f"{classes} {tag_id}"

        if NOISE_CLASS_PATTERN.search(blob):
            if tag.name not in ("article", "main"):
                tag.decompose()

    return deduplicate_preserve_order(_bs4_extract_lines(soup))


# =========================
# DEDUPLICATION
# =========================
def deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Supprime les doublons en gardant l'ordre."""
    seen = set()
    result = []
    for item in items:
        key = normalize_extracted_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


# =========================
# EVALUATION SIMPLE DE LA QUALITE
# =========================
def is_suspect_noise_line(text: str) -> bool:
    """Détecte des lignes qui ressemblent encore à du bruit."""
    if not text:
        return False

    t = normalize_extracted_text(text).lower()

    suspect_patterns = [
        "lire aussi",
        "à lire aussi",
        "voir aussi",
        "en savoir plus",
        "articles liés",
        "contenus liés",
        "sur le même sujet",
        "partager",
        "imprimer",
        "newsletter",
        "publicité",
        "retour en haut",
        "choisir la langue",
        "page d'accueil",
        "accès direct au contenu",
        "accès direct au menu principal",
        "get the latest updates",
        "special offers",
        "top stories",
        "upcoming events",
        "sign up",
        "subscribe",
        "latest from",
        "from mit technology review",
    ]

    return any(p in t for p in suspect_patterns)


def evaluate_extraction_quality(meta: dict, content_lines: List[str]) -> dict:
    """Produit quelques indicateurs simples sur la qualité du résultat."""
    nb_lines = len(content_lines)
    nb_paragraphs = sum(1 for line in content_lines if line.startswith("[PARAGRAPHE]"))

    heading_lines = [
        line for line in content_lines
        if line.startswith("[TITRE_ARTICLE]") or
           line.startswith("[H1]") or
           line.startswith("[H2]") or
           line.startswith("[H3]") or
           line.startswith("[H4]") or
           line.startswith("[INTERTITRE]")
    ]

    nb_headings = len(heading_lines)
    nb_lists = sum(1 for line in content_lines if line.startswith("[LISTE]"))

    total_text_length = sum(len(normalize_extracted_text(line)) for line in content_lines)

    suspicious_noise_lines = sum(
        1 for line in content_lines if is_suspect_noise_line(line)
    )

    suspicious_heading_count = sum(
        1 for line in heading_lines if is_suspect_noise_line(line)
    )

    has_title = bool(meta.get("title")) or any(
        line.startswith("[TITRE_ARTICLE]") for line in content_lines
    )

    probable_heading_paragraphs = sum(
        1 for line in content_lines
        if line.startswith("[PARAGRAPHE]") and looks_like_heading_candidate(
            line.replace("[PARAGRAPHE]", "", 1).strip()
        )
    )

    if total_text_length >= 3000 and nb_paragraphs >= 3:
        content_quality = "bon"
    elif total_text_length >= 1200 and nb_paragraphs >= 2:
        content_quality = "moyen"
    else:
        content_quality = "faible"

    useful_headings = max(0, nb_headings - suspicious_heading_count)
    effective_headings = useful_headings + probable_heading_paragraphs

    if nb_paragraphs >= 2 and effective_headings >= 1:
        structure_quality = "correcte"
    elif nb_paragraphs >= 1:
        structure_quality = "partielle"
    else:
        structure_quality = "faible"

    total_noise_signals = suspicious_noise_lines + suspicious_heading_count

    if total_noise_signals == 0:
        noise_level = "faible"
    elif total_noise_signals <= 2:
        noise_level = "moyen"
    else:
        noise_level = "élevé"

    return {
        "has_title": has_title,
        "nb_lines": nb_lines,
        "nb_paragraphs": nb_paragraphs,
        "nb_headings": nb_headings,
        "nb_lists": nb_lists,
        "total_text_length": total_text_length,
        "suspicious_noise_lines": suspicious_noise_lines,
        "suspicious_heading_count": suspicious_heading_count,
        "content_quality": content_quality,
        "structure_quality": structure_quality,
        "noise_level": noise_level,
        "probable_heading_paragraphs": probable_heading_paragraphs,
    }


# =========================
# CONSTRUCTION DU TEXTE FINAL
# =========================
def build_output_text(file_name: str, meta: dict, content_lines: List[str], quality: dict) -> str:
    """Construit le texte final."""
    parts: List[str] = []

    parts.append(f"NOM_DE_LA_PAGE : {file_name}")

    if meta.get("title"):
        parts.append(f"[TITLE] {meta['title']}")

    if meta.get("summary"):
        parts.append(f"[RESUME] {meta['summary']}")

    if content_lines:
        parts.append("")
        parts.append("CONTENU :")
        parts.extend(content_lines)

    parts.append("")
    parts.append("QUALITE :")
    parts.append(f"- contenu : {quality['content_quality']}")
    parts.append(f"- structure : {quality['structure_quality']}")
    parts.append(f"- bruit : {quality['noise_level']}")
    parts.append(f"- titre_present : {'oui' if quality['has_title'] else 'non'}")
    parts.append(f"- nb_lignes : {quality['nb_lines']}")
    parts.append(f"- nb_paragraphes : {quality['nb_paragraphs']}")
    parts.append(f"- nb_titres : {quality['nb_headings']}")
    parts.append(f"- nb_listes : {quality['nb_lists']}")
    parts.append(f"- longueur_texte : {quality['total_text_length']}")
    parts.append(f"- lignes_bruit_suspectes : {quality['suspicious_noise_lines']}")
    parts.append(f"- titres_suspects : {quality['suspicious_heading_count']}")
    parts.append(f"- paragraphes_type_intertitre : {quality['probable_heading_paragraphs']}")

    return "\n".join(parts).strip() + "\n"

# =========================
# TRAITEMENT D'UN FICHIER
# =========================
def process_html_file(html_path: Path, output_dir: Path) -> None:
    """Traite un fichier HTML et génère son fichier texte."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"Erreur lecture {html_path.name} : {exc}")
        return

    # Métadonnées avant pré-traitement
    meta = extract_meta_from_html(html)

    # Pré-traitement du HTML
    html_clean = preprocess_html(html)

    # Tentative 1 : trafilatura précis
    xml_content = trafilatura_extract_xml(html_clean, favor_precision=True)
    content_lines = parse_trafilatura_xml(xml_content) if xml_content else []

    # Tentative 2 : trafilatura plus souple
    if len(content_lines) < 3:
        xml_content2 = trafilatura_extract_xml(html_clean, favor_precision=False)
        lines2 = parse_trafilatura_xml(xml_content2) if xml_content2 else []
        if len(lines2) > len(content_lines):
            content_lines = lines2

    # Tentative 3 : fallback BS4 sur HTML nettoyé
    if len(content_lines) < 3:
        fallback_lines = fallback_extract_with_bs4(html_clean)
        if len(fallback_lines) > len(content_lines):
            content_lines = fallback_lines

    # Tentative 4 : fallback BS4 sur HTML original
    if len(content_lines) < 3:
        fallback_orig = fallback_extract_with_bs4(html)
        if len(fallback_orig) > len(content_lines):
            content_lines = fallback_orig

    quality = evaluate_extraction_quality(meta, content_lines)
    output_text = build_output_text(html_path.name, meta, content_lines, quality)

    output_file = output_dir / f"{safe_filename(html_path.name)}.txt"
    try:
        output_file.write_text(output_text, encoding="utf-8")
        print(f"OK : {html_path.name} -> {output_file.name}")
    except Exception as exc:
        print(f"Erreur écriture {output_file.name} : {exc}")


# =========================
# POINT D'ENTREE
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_directory(OUTPUT_DIR)

    html_files = sorted(
        list(SOURCE_DIR.glob("*.html")) +
        list(SOURCE_DIR.glob("*.htm"))
    )

    if not html_files:
        print("Aucun fichier HTML trouvé.")
        return

    for html_file in html_files:
        process_html_file(html_file, OUTPUT_DIR)

    print("Traitement terminé.")


if __name__ == "__main__":
    main()