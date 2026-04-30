"""
crawler.py — Web news crawler using Playwright
Stratégie : listing pages → pagination → articles (pas de BFS généraliste)
"""

import asyncio
import json
import logging
import os
import re
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

# Supprime les messages internes de Playwright (navigation, networkidle…)
logging.getLogger("playwright").setLevel(logging.ERROR)

import yaml
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE          = Path(r"C:/PYTHON/.entree/SitesSources/sites_actus_labos.yaml")
OUTPUT_DIR          = Path(r"C:/PYTHON/.data/Resultatscrawling")
JOURNAL_FILE        = Path(r"C:/PYTHON/.data/Resultatscrawling/journal.json")
DELAY_MIN           = 2.0   # secondes, délai minimum entre deux requêtes
DELAY_MAX           = 5.0   # secondes, délai maximum entre deux requêtes
MAX_LISTING_PAGES   = 10    # nb max de pages de listing visitées par start_url (0 = illimité)
MAX_ARTICLES_PER_SITE = 50  # nb max d'articles sauvegardés par site (0 = illimité)
RESPECT_ROBOTS      = True
HEADLESS            = False # navigateur visible pour résoudre les anti-bots manuellement
CURRENT_YEAR_ONLY   = True  # ne retenir que les articles de l'année en cours
SAVE_IF_NO_DATE     = False # sauvegarder si aucune date n'est détectable

# Regex et format date globaux — surchargés par la section "defaults" du YAML
_DEFAULT_DATE_REGEX  = r"(\d{2}/\d{2}/20\d{2})"   # format DD/MM/YYYY
_DEFAULT_DATE_FORMAT = "%d/%m/%Y"

EXCLUDED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".avi", ".zip", ".tar", ".gz", ".exe",
    ".css", ".js", ".xml", ".json", ".ico", ".woff", ".woff2",
    ".ttf", ".eot", ".rss",
}

# Patterns exclus pour les liens candidats articles (pas pour la pagination)
EXCLUDED_PATH_PATTERNS = [
    r"/tag/", r"/tags/", r"/author/", r"/auteur/", r"/auteurs/",
    r"/category/", r"/categorie/", r"/feed/?$", r"/rss/?$",
    r"/search[/?]", r"/login", r"/register", r"/cart", r"/panier",
    r"/account/", r"/cdn-cgi/", r"[?&]s=",
]

# Patterns de détection de pagination (URL)
_PAGINATION_URL_PATTERNS = [
    r"/page/\d+/?$",
    r"[?&]page=\d+",
    r"[?&]paged=\d+",
    r"[?&]p=\d+$",
    r"[?&]offset=\d+",
    r"[?&]start=\d+",
    r"[?&]from=\d+",
    r"/\d+/?$",         # ex: /actualites/2/ sur certains sites
]

ANTIBOT_SIGNALS = [
    "cf-browser-verification", "challenge-form", "cf_chl_opt",
    "access denied", "attention required",
    "enable javascript and cookies",
    "please verify you are a human",
    "ddos protection by cloudflare",
    "please complete the security check",
    "just a moment",
]

# ─────────────────────────────────────────────────────────────────────────────
# SITE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SiteConfig:
    name:                str
    start_urls:          list[str]
    article_url_pattern: str | None = None  # regex : si match → c'est un article
    pagination_pattern:  str | None = None  # regex : si match → c'est une page de pagination
    date_selector:       str | None = None  # sélecteur CSS pour trouver la date
    date_regex:          str | None = None  # regex pour extraire la date du texte trouvé
    date_format:         str | None = None  # format strptime (ex. "%d/%m/%Y")
    disable_nav_filter:  bool       = False # conservé pour compatibilité YAML

def load_sites() -> list[SiteConfig]:
    """Charge le fichier YAML et retourne la liste des SiteConfig."""
    global _DEFAULT_DATE_REGEX, _DEFAULT_DATE_FORMAT

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults", {})
    if "date_regex" in defaults:
        _DEFAULT_DATE_REGEX = defaults["date_regex"]
    if "date_format" in defaults:
        _DEFAULT_DATE_FORMAT = defaults["date_format"]

    sites = []
    for s in data.get("sites", []):
        start_urls = s.get("start_urls", [])
        if not start_urls:
            continue
        sites.append(SiteConfig(
            name                = s.get("name", start_urls[0]),
            start_urls          = start_urls,
            article_url_pattern = s.get("article_url_pattern"),
            pagination_pattern  = s.get("pagination_pattern"),
            date_selector       = s.get("date_selector"),
            date_regex          = s.get("date_regex"),
            date_format         = s.get("date_format"),
            disable_nav_filter  = s.get("disable_nav_filter", False),
        ))
    return sites

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
def load_journal() -> dict:
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_journal(journal: dict):
    with open(JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)

def clear_journal():
    if JOURNAL_FILE.exists():
        os.remove(JOURNAL_FILE)
        log("Journal vidé.")

# ─────────────────────────────────────────────────────────────────────────────
# URL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    return url.split("#")[0].rstrip("/")

def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def is_same_domain(url: str, base_domain: str) -> bool:
    host = get_domain(url).removeprefix("www.")
    base = base_domain.lower().removeprefix("www.")
    return host == base

def is_valid_link(url: str) -> bool:
    """Filtre les extensions non-HTML et les chemins manifestement non-éditoriaux."""
    path = urlparse(url).path.lower()
    for ext in EXCLUDED_EXTENSIONS:
        if path.endswith(ext):
            return False
    for pattern in EXCLUDED_PATH_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return False
    return True

def generate_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path   = parsed.path.strip("/").replace("/", "__")
    query  = re.sub(r"[^\w]", "_", parsed.query)[:40] if parsed.query else ""
    parts  = [domain]
    if path:
        parts.append(path)
    if query:
        parts.append(query)
    name = "__".join(parts)
    name = re.sub(r"[^\w\-_.]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:180] + ".html"

# ─────────────────────────────────────────────────────────────────────────────
# ROBOTS.TXT
# ─────────────────────────────────────────────────────────────────────────────
_robots_cache: dict = {}

def is_allowed_by_robots(url: str) -> bool:
    if not RESPECT_ROBOTS:
        return True
    parsed     = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    if robots_url not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            _robots_cache[robots_url] = rp
        except Exception:
            _robots_cache[robots_url] = None
    rp = _robots_cache[robots_url]
    return rp.can_fetch("*", url) if rp else True

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-BOT
# ─────────────────────────────────────────────────────────────────────────────
async def detect_antibot(page) -> bool:
    try:
        content = (await page.content()).lower()
        title   = (await page.title()).lower()
        return any(s in content or s in title for s in ANTIBOT_SIGNALS)
    except Exception:
        return False

async def wait_for_human():
    log("ANTI-BOT DETECTE — Résolvez le challenge dans le navigateur.")
    log("Appuyez sur Entrée pour continuer...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input)

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION (partagée listing + article)
# ─────────────────────────────────────────────────────────────────────────────
async def navigate_page(page, url: str):
    """Navigue vers url avec fallback et gestion anti-bot. Retourne la response."""
    try:
        response = await page.goto(url, wait_until="networkidle", timeout=30000)
        # Petite pause de stabilisation même après networkidle :
        # certains sites font du rendu JS en cascade après le dernier réseau.
        await asyncio.sleep(1.0)
    except Exception:
        log(f"  -> networkidle timeout, repli sur domcontentloaded")
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)   # délai plus long pour laisser le JS se terminer

    if await detect_antibot(page):
        await wait_for_human()
        await asyncio.sleep(2)
        await page.wait_for_load_state("networkidle", timeout=60000)

    return response

# ─────────────────────────────────────────────────────────────────────────────
# DATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _parse_year(value: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", str(value))
    return int(m.group(1)) if m else None

def _year_from_text(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return _parse_year(m.group(0)) if m else None

def extract_article_year(html: str, url: str, site_cfg: SiteConfig) -> int | None:
    soup = BeautifulSoup(html, "lxml")

    # 1. Sélecteur CSS spécifique au site
    if site_cfg.date_selector:
        el = soup.select_one(site_cfg.date_selector)
        if el:
            year = _year_from_text(el.get_text(), site_cfg.date_regex or _DEFAULT_DATE_REGEX)
            if year:
                return year

    # 2. Regex texte spécifique au site (sans sélecteur CSS)
    if site_cfg.date_regex and not site_cfg.date_selector:
        year = _year_from_text(soup.get_text(" ", strip=True), site_cfg.date_regex)
        if year:
            return year

    # 3. Heuristiques sémantiques standard
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            for key in ("datePublished", "dateCreated", "dateModified"):
                if data.get(key):
                    year = _parse_year(data[key])
                    if year:
                        return year
        except Exception:
            pass

    og = soup.find("meta", property="article:published_time")
    if og:
        year = _parse_year(og.get("content", ""))
        if year:
            return year

    for name in ("date", "pubdate", "publish_date", "DC.date", "article.published"):
        meta = soup.find("meta", attrs={"name": name})
        if meta:
            year = _parse_year(meta.get("content", ""))
            if year:
                return year

    time_el = soup.find("time", attrs={"datetime": True})
    if time_el:
        year = _parse_year(time_el["datetime"])
        if year:
            return year

    # 4. Pattern dans l'URL : /2026/
    m = re.search(r"/(20\d{2})[/\-]", url)
    if m:
        return int(m.group(1))

    # 5. Fallback global : regex DD/MM/YYYY dans le texte de la page
    year = _year_from_text(soup.get_text(" ", strip=True), _DEFAULT_DATE_REGEX)
    if year:
        return year

    # 6. Année dans le slug URL sans frontière stricte (ex: "titre-2026a1000x")
    m = re.search(r'\b(20\d{2})', url)
    if m:
        return int(m.group(1))

    return None

# ─────────────────────────────────────────────────────────────────────────────
# NEWS PAGE DETECTION (fallback quand pas de article_url_pattern)
# ─────────────────────────────────────────────────────────────────────────────
_DATE_CLASS   = re.compile(r"\bdate\b|pubdate|published|timestamp|posted", re.I)
_AUTHOR_CLASS = re.compile(r"\bauthor\b|byline|auteur", re.I)
_BODY_CLASS   = re.compile(r"article.?body|post.?content|entry.?content|article.?content|article.?text", re.I)

def is_news_page(html: str, url: str, site_cfg: SiteConfig) -> bool:
    if site_cfg.article_url_pattern:
        return bool(re.search(site_cfg.article_url_pattern, url, re.IGNORECASE))
    soup  = BeautifulSoup(html, "lxml")
    score = 0
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data  = json.loads(script.string or "")
            types = data.get("@type", "")
            if isinstance(types, list):
                types = " ".join(types)
            if any(t in types for t in ["NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle"]):
                score += 4
        except Exception:
            pass
    og = soup.find("meta", property="og:type")
    if og and "article" in og.get("content", "").lower():
        score += 3
    if soup.find("article"):
        score += 2
    if soup.find("time") or soup.find(class_=_DATE_CLASS):
        score += 2
    if soup.find(rel="author") or soup.find(class_=_AUTHOR_CLASS):
        score += 1
    if soup.find("meta", attrs={"name": "author"}):
        score += 1
    if soup.find(class_=_BODY_CLASS):
        score += 2
    return score >= 3

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION DES LIENS
# ─────────────────────────────────────────────────────────────────────────────
def is_article_url(url: str, site_cfg: SiteConfig) -> bool:
    """Retourne True si l'URL est un article selon le pattern du site."""
    if site_cfg.article_url_pattern:
        return bool(re.search(site_cfg.article_url_pattern, url, re.IGNORECASE))
    # Sans pattern : tout lien valide est candidat article (filtré par HTML ensuite)
    return is_valid_link(url)

def is_pagination_url(url: str, site_cfg: SiteConfig) -> bool:
    """Retourne True si l'URL est une page de pagination."""
    if site_cfg.pagination_pattern:
        return bool(re.search(site_cfg.pagination_pattern, url, re.IGNORECASE))
    for pattern in _PAGINATION_URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DES LIENS D'UNE PAGE DE LISTING
# ─────────────────────────────────────────────────────────────────────────────
async def extract_listing_links(
    page, base_url: str, base_domain: str, site_cfg: SiteConfig
) -> tuple[list[str], list[str]]:
    """
    Depuis une page de listing, retourne (article_urls, pagination_urls).
    Tous les liens sont collectés sans filtre nav (on veut tous les articles).
    """
    try:
        hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    except Exception:
        log(f"  -> [DEBUG] eval_on_selector_all a échoué")
        return [], []

    seen              = set()
    articles          = []
    pagination        = []
    n_off_domain      = 0
    n_no_pattern      = 0

    for href in hrefs:
        url = normalize_url(urljoin(base_url, href))
        if url in seen or not url.startswith("http"):
            continue
        seen.add(url)

        if not is_same_domain(url, base_domain):
            n_off_domain += 1
            continue

        if is_pagination_url(url, site_cfg):
            pagination.append(url)
        elif is_article_url(url, site_cfg) and is_allowed_by_robots(url):
            articles.append(url)
        else:
            n_no_pattern += 1

    log(f"  -> [DEBUG] {len(hrefs)} liens bruts | "
        f"{n_off_domain} hors-domaine | "
        f"{n_no_pattern} sans pattern | "
        f"{len(articles)} articles | {len(pagination)} paginations")
    return articles, pagination

# ─────────────────────────────────────────────────────────────────────────────
# VISITE D'UNE PAGE DE LISTING
# ─────────────────────────────────────────────────────────────────────────────
async def visit_listing(
    page, url: str, base_domain: str,
    site_cfg: SiteConfig, journal: dict
) -> tuple[list[str], list[str]]:
    """
    Navigue vers une page de listing.
    Retourne (article_urls, pagination_urls) ou ([], []) en cas d'erreur.
    """
    journal_key = f"__listing__{url}"

    # Reprise : listing déjà visité dans une session précédente ET ayant produit des résultats.
    # Si le run précédent a stocké 0 articles + 0 pagination (possible erreur de timing JS),
    # on revisite la page pour ne pas rater des articles.
    if journal_key in journal:
        data = journal[journal_key]
        cached_articles    = data.get("articles", [])
        cached_pagination  = data.get("pagination", [])
        if cached_articles or cached_pagination:
            log(f"[LISTING SKIP] {url}  ({len(cached_articles)} articles en cache)")
            return cached_articles, cached_pagination
        else:
            log(f"[LISTING RETRY] {url}  (résultat vide en cache → nouvelle tentative)")

    log(f"[LISTING] {url}")
    try:
        response = await navigate_page(page, url)

        if response and response.status in (403, 404, 410, 429, 503):
            log(f"  -> HTTP {response.status} — ignoré")
            journal[journal_key] = {"articles": [], "pagination": []}
            save_journal(journal)
            return [], []

        articles, pagination = await extract_listing_links(page, url, base_domain, site_cfg)
        log(f"  -> {len(articles)} article(s), {len(pagination)} page(s) suivante(s)")

        # Ne journaliser que si des résultats ont été trouvés :
        # un résultat vide ne doit pas "geler" la page pour les runs suivants.
        if articles or pagination:
            journal[journal_key] = {"articles": articles, "pagination": pagination}
            save_journal(journal)
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        return articles, pagination

    except Exception as e:
        log(f"  -> Erreur listing : {e}")
        # Ne pas journaliser les erreurs de listing non-HTTP (timeout, JS…) :
        # elles peuvent être transitoires.
        return [], []

# ─────────────────────────────────────────────────────────────────────────────
# VISITE D'UN ARTICLE
# ─────────────────────────────────────────────────────────────────────────────
async def visit_article(
    page, url: str, base_domain: str,
    site_cfg: SiteConfig, journal: dict, stats: dict
):
    """Navigue vers un article, détecte l'année, sauvegarde si pertinent."""
    log(f"[ARTICLE] {url}")
    try:
        response = await navigate_page(page, url)

        if response and response.status in (403, 404, 410, 429, 503):
            log(f"  -> HTTP {response.status} — ignoré")
            journal[url] = {"status": "error", "code": response.status}
            stats["errors"] += 1
            save_journal(journal)
            return

        html = await page.content()

        # Vérification HTML pour les sites sans article_url_pattern
        if not site_cfg.article_url_pattern and not is_news_page(html, url, site_cfg):
            log(f"  -> pas un article (heuristiques HTML)")
            journal[url] = {"status": "not_article"}
            stats["visited"] = stats.get("visited", 0) + 1
            save_journal(journal)
            await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            return

        year = extract_article_year(html, url, site_cfg)
        if CURRENT_YEAR_ONLY and year != datetime.now().year \
                and not (year is None and SAVE_IF_NO_DATE):
            label = str(year) if year else "date inconnue"
            log(f"  -> ignoré ({label}, hors année en cours)")
            journal[url] = {"status": "skipped_year", "year": year}
            stats["skipped_year"] = stats.get("skipped_year", 0) + 1
        else:
            filename = generate_filename(url)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / filename).write_text(html, encoding="utf-8")
            label = str(year) if year else "année inconnue"
            log(f"  -> SAUVEGARDE ({label}) : {filename}")
            journal[url] = {"status": "saved", "file": filename}
            stats["saved"] += 1

        save_journal(journal)
        await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    except Exception as e:
        log(f"  -> Erreur : {e}")
        journal[url] = {"status": "error", "message": str(e)}
        stats["errors"] += 1
        save_journal(journal)

# ─────────────────────────────────────────────────────────────────────────────
# CRAWL SITE
# ─────────────────────────────────────────────────────────────────────────────
async def crawl_site(page, site_cfg: SiteConfig, journal: dict, stats: dict):
    first_url   = normalize_url(site_cfg.start_urls[0])
    base_domain = get_domain(first_url)

    log(f"\n{'─' * 60}")
    log(f"SITE : {site_cfg.name}")
    for u in site_cfg.start_urls:
        log(f"  start : {u}")
    log(f"{'─' * 60}")

    # ── Phase 1 : découverte des articles depuis les pages de listing ─────────
    all_article_urls: list[str] = []
    seen_articles:    set[str]  = set()

    for start_url in site_cfg.start_urls:
        start_url        = normalize_url(start_url)
        visited_listings = set()
        listing_queue    = [start_url]

        while listing_queue:
            listing_url = normalize_url(listing_queue.pop(0))
            if listing_url in visited_listings:
                continue
            visited_listings.add(listing_url)

            # Limite de pages de listing
            listing_count = len(visited_listings)
            if MAX_LISTING_PAGES and listing_count > MAX_LISTING_PAGES:
                log(f"  -> limite de {MAX_LISTING_PAGES} pages de listing atteinte.")
                break

            articles, pagination = await visit_listing(
                page, listing_url, base_domain, site_cfg, journal
            )

            for url in articles:
                if url not in seen_articles:
                    seen_articles.add(url)
                    all_article_urls.append(url)

            for purl in pagination:
                purl = normalize_url(purl)
                if purl not in visited_listings:
                    listing_queue.append(purl)

    log(f"  {len(all_article_urls)} article(s) découvert(s) au total")

    # ── Phase 2 : visite des articles ────────────────────────────────────────
    site_saved = 0

    for article_url in all_article_urls:
        if MAX_ARTICLES_PER_SITE and site_saved >= MAX_ARTICLES_PER_SITE:
            log(f"  -> limite de {MAX_ARTICLES_PER_SITE} articles atteinte pour ce site.")
            break

        if article_url in journal:
            log(f"[SKIP] {article_url}")
            if journal[article_url].get("status") == "saved":
                site_saved += 1
            continue

        saved_before = stats["saved"]
        await visit_article(page, article_url, base_domain, site_cfg, journal, stats)
        if stats["saved"] > saved_before:
            site_saved += 1

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(stats: dict, journal: dict):
    log(f"\n{'=' * 60}")
    log("RAPPORT FINAL")
    log(f"{'=' * 60}")
    log(f"Pages visitées       : {stats.get('visited', 0)}")
    log(f"Articles sauvegardés : {stats['saved']}")
    log(f"Articles hors année  : {stats.get('skipped_year', 0)}")
    log(f"Erreurs              : {stats['errors']}")
    log(f"Répertoire de sortie : {OUTPUT_DIR}/")

    saved = [(u, d["file"]) for u, d in journal.items()
             if isinstance(d, dict) and d.get("status") == "saved"]
    if saved:
        log(f"\n{len(saved)} article(s) :")
        for url, filename in saved:
            log(f"  {filename}")
            log(f"    {url}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    sites   = load_sites()
    journal = load_journal()
    stats   = {"visited": 0, "saved": 0, "errors": 0}

    log(f"{len(sites)} site(s) chargé(s) depuis {INPUT_FILE.name}")

    if journal:
        already = sum(1 for v in journal.values()
                      if isinstance(v, dict) and v.get("status") in ("saved", "skipped_year"))
        log(f"Reprise détectée : {already} articles déjà traités dans le journal.")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        for site_cfg in sites:
            await crawl_site(page, site_cfg, journal, stats)

        await browser.close()

    generate_report(stats, journal)
    clear_journal()

if __name__ == "__main__":
    asyncio.run(main())
