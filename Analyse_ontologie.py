#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classer un texte phrase par phrase avec une ontologie YAML (structure plate),
SANS paramètres en ligne de commande : tout est piloté par des variables.

Sorties (dans OUTDIR) :
  - phrases_classified.csv
  - category_stats.csv
  - results.json

Ajout :
- prise en charge d'une section YAML "synonyms" :
  - canonical: "scanner"
    variants: ["tdm", "tomodensitométrie"]
- le texte est normalisé puis les synonymes sont appliqués AVANT le matching.
"""

# =========================================================
# BIBLIOTHEQUES
# =========================================================
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from datetime import datetime


# =========================================================
# PARAMÈTRES (à modifier ici)
# =========================================================

ONTOLOGY_FILE = Path("D:/CODING/.params/ontologie_medicale_fr.yaml")  # ontologie YAML (structure plate)
#ONTOLOGY_FILE = Path("D:/CODING/.params/ontologie_medicale_en.yaml")  # ontologie YAML (structure plate)

INPUT_FILE = Path("D:/CODING/.params/entree.txt")                  # fichier texte à classer (txt/md)
OUTDIR = Path("D:/CODING/.data/")                                  # dossier de sortie

MIN_SCORE = 1            # nb minimal de mots-clés trouvés pour garder une catégorie
ALLOW_MULTI = True       # True = une phrase peut avoir plusieurs catégories
DOMINANT_ONLY = False    # True = force 1 seule catégorie par phrase (la meilleure)
MAX_PHRASES = 0          # 0 = pas de limite, sinon ex: 200

# Si aucune catégorie ne match une phrase, on peut la marquer (optionnel)
ADD_UNCLASSIFIED_ROW = True
UNCLASSIFIED_ID = "unclassified"
UNCLASSIFIED_LABEL = "Non classé"


# =========================================================
# Modèles simples
# =========================================================

@dataclass
class Category:
    id: str
    label: str
    keywords: List[str]


# =========================================================
# Normalisation texte
# =========================================================

_PUNCT_RE = re.compile(r"[^\w\sàâäçéèêëîïôöùûüÿñæœ'-]+", flags=re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# Synonymes (AJOUT)
# =========================================================

def load_synonyms(yaml_root: dict) -> List[Tuple[str, List[str]]]:
    """
    Lit la clé 'synonyms' du YAML et renvoie une liste :
      [(canonical, [variant1, variant2, ...]), ...]
    Tout est normalisé (minuscules / ponctuation / espaces).
    """
    syn = yaml_root.get("synonyms", []) or []
    out: List[Tuple[str, List[str]]] = []
    for item in syn:
        if not isinstance(item, dict):
            continue
        canonical = normalize_text(str(item.get("canonical", "")))
        variants = item.get("variants", []) or []
        if not isinstance(variants, list):
            continue
        variants_norm = [normalize_text(str(v)) for v in variants]
        variants_norm = [v for v in variants_norm if v]
        if canonical and variants_norm:
            out.append((canonical, variants_norm))
    return out


def apply_synonyms(text_norm: str, synonyms: List[Tuple[str, List[str]]]) -> str:
    """
    Remplace dans text_norm (déjà normalisé) les variantes par la forme canonical,
    en remplaçant uniquement des mots / expressions entiers (pas au milieu des mots).
    """
    if not text_norm or not synonyms:
        return text_norm

    for canonical, variants in synonyms:
        for v in variants:
            # tolère espaces multiples dans les expressions
            v_regex = re.escape(v).replace(r"\ ", r"\s+")
            text_norm = re.sub(rf"\b{v_regex}\b", canonical, text_norm, flags=re.IGNORECASE)

    text_norm = re.sub(r"\s+", " ", text_norm).strip()
    return text_norm


# =========================================================
# Chargement ontologie
# =========================================================

def load_ontology_yaml(path: Path) -> Tuple[List[Category], Dict, List[Tuple[str, List[str]]]]:
    yaml_root = yaml.safe_load(path.read_text(encoding="utf-8"))
    print("{} - Chargement Ontologie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    if not isinstance(yaml_root, dict) or "ontology" not in yaml_root:
        raise ValueError("Fichier ontologie invalide : clé 'ontology' manquante.")

    cats: List[Category] = []
    for item in yaml_root["ontology"]:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id", "")).strip()
        label = str(item.get("label", cid)).strip()
        keywords = item.get("keywords", [])
        if not cid:
            continue
        if not isinstance(keywords, list):
            raise ValueError(f"Keywords invalides pour {cid} : doit être une liste.")
        keywords_clean = [str(k).strip() for k in keywords if str(k).strip()]
        cats.append(Category(id=cid, label=label, keywords=keywords_clean))

    meta = yaml_root.get("meta", {})
    synonyms = load_synonyms(yaml_root)
    return cats, meta, synonyms


# =========================================================
# Découpage en phrases
# =========================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    print("{} - Séparation des phrases".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 2:
            continue
        sentences.append(p)
    return sentences


# =========================================================
# Matching keywords
# =========================================================

def build_keyword_patterns(categories: List[Category]) -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    print("{} - Keyword patterns".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for c in categories:
        compiled: List[re.Pattern] = []
        for kw in c.keywords:
            kw_norm = normalize_text(kw)
            if not kw_norm:
                continue
            kw_regex = re.escape(kw_norm).replace(r"\ ", r"\s+")
            compiled.append(re.compile(rf"\b{kw_regex}\b", flags=re.IGNORECASE))
        patterns[c.id] = compiled
    return patterns


def score_sentence(sentence_norm: str,
                   categories: List[Category],
                   patterns: Dict[str, List[re.Pattern]]) -> List[Dict]:
    results: List[Dict] = []
    for c in categories:
        matched: List[str] = []
        pats = patterns.get(c.id, [])
        for kw, pat in zip(c.keywords, pats):
            if pat.search(sentence_norm):
                matched.append(kw)
        score = len(matched)
        if score > 0:
            results.append({
                "category_id": c.id,
                "category_label": c.label,
                "score": score,
                "matched_keywords": matched
            })
    results.sort(key=lambda x: (-x["score"], x["category_label"]))
    return results


# =========================================================
# Pipeline principal
# =========================================================

def classify_text(
    text: str,
    categories: List[Category],
    synonyms: List[Tuple[str, List[str]]],   # <-- AJOUT
    min_score: int,
    allow_multi: bool,
    dominant_only: bool,
    max_phrases: int,
    add_unclassified_row: bool
) -> Dict:
    patterns = build_keyword_patterns(categories)
    sentences = split_sentences(text)
    if max_phrases and max_phrases > 0:
        sentences = sentences[:max_phrases]

    classified_rows: List[Dict] = []

    category_counts = {c.id: 0 for c in categories}
    category_labels = {c.id: c.label for c in categories}

    unclassified_count = 0
    total_sentences = len(sentences)

    for idx, s in enumerate(sentences, start=1):
        # 1) normalisation
        s_norm = normalize_text(s)
        # 2) synonymes (AJOUT) : on remplace les variantes par canonical
        s_norm_syn = apply_synonyms(s_norm, synonyms)

        # matching sur la version "après synonymes"
        matches = score_sentence(s_norm_syn, categories, patterns)
        matches = [m for m in matches if m["score"] >= min_score]

        dominant = matches[0] if matches else None

        if dominant_only:
            chosen = [dominant] if dominant else []
        else:
            chosen = matches if allow_multi else ([dominant] if dominant else [])

        if not chosen and add_unclassified_row:
            unclassified_count += 1
            chosen_ids = UNCLASSIFIED_ID
            chosen_labels = UNCLASSIFIED_LABEL
            chosen_scores = "0"
            chosen_keywords = ""
            dom_id = ""
            dom_label = ""
            dom_score = 0
        else:
            for m in chosen:
                category_counts[m["category_id"]] += 1

            chosen_ids = ";".join([m["category_id"] for m in chosen])
            chosen_labels = ";".join([m["category_label"] for m in chosen])
            chosen_scores = ";".join([str(m["score"]) for m in chosen])
            chosen_keywords = ";".join([",".join(m["matched_keywords"]) for m in chosen])

            dom_id = dominant["category_id"] if dominant else ""
            dom_label = dominant["category_label"] if dominant else ""
            dom_score = dominant["score"] if dominant else 0

        classified_rows.append({
            "sentence_index": idx,
            "sentence": s,
            "sentence_normalized": s_norm,
            "sentence_normalized_after_synonyms": s_norm_syn,  # <-- AJOUT (utile pour debug)
            "dominant_category_id": dom_id,
            "dominant_category_label": dom_label,
            "dominant_score": dom_score,
            "all_categories": chosen_ids,
            "all_labels": chosen_labels,
            "all_scores": chosen_scores,
            "all_matched_keywords": chosen_keywords
        })

    stats_rows: List[Dict] = []
    for c in categories:
        occ = category_counts[c.id]
        pct = (occ / total_sentences) if total_sentences else 0.0
        stats_rows.append({
            "category_id": c.id,
            "category_label": category_labels[c.id],
            "occurrences": occ,
            "percent_of_sentences": pct
        })

    if add_unclassified_row:
        pct_u = (unclassified_count / total_sentences) if total_sentences else 0.0
        stats_rows.append({
            "category_id": UNCLASSIFIED_ID,
            "category_label": UNCLASSIFIED_LABEL,
            "occurrences": unclassified_count,
            "percent_of_sentences": pct_u
        })

    stats_rows.sort(key=lambda x: (-x["occurrences"], x["category_label"]))

    return {
        "total_sentences": total_sentences,
        "settings": {
            "min_score": min_score,
            "allow_multi": allow_multi,
            "dominant_only": dominant_only,
            "max_phrases": max_phrases,
            "add_unclassified_row": add_unclassified_row,
            "synonyms_count": len(synonyms)  # <-- AJOUT
        },
        "classified_rows": classified_rows,
        "stats_rows": stats_rows
    }


def write_outputs(result: Dict, outdir: Path) -> Dict[str, Path]:
    print("{} - Ecriture de la sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    outdir.mkdir(parents=True, exist_ok=True)

    df_phrases = pd.DataFrame(result["classified_rows"])
    df_stats = pd.DataFrame(result["stats_rows"])

    phrases_csv = outdir / "phrases_classified.csv"
    stats_csv = outdir / "category_stats.csv"
    json_path = outdir / "results.json"

    df_phrases.to_csv(phrases_csv, index=False, encoding="utf-8-sig")
    df_stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"phrases_csv": phrases_csv, "stats_csv": stats_csv, "json": json_path}


# =========================================================
# main
# =========================================================

def main():
    print("{} - Debut du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {INPUT_FILE}")
    if not ONTOLOGY_FILE.exists():
        raise FileNotFoundError(f"Ontologie introuvable: {ONTOLOGY_FILE}")

    text = INPUT_FILE.read_text(encoding="utf-8", errors="ignore")

    # <-- MODIF : on récupère aussi synonyms
    categories, meta, synonyms = load_ontology_yaml(ONTOLOGY_FILE)
    print("{} - Synonymes chargés: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), len(synonyms)))

    result = classify_text(
        text=text,
        categories=categories,
        synonyms=synonyms,  # <-- AJOUT
        min_score=MIN_SCORE,
        allow_multi=ALLOW_MULTI,
        dominant_only=DOMINANT_ONLY,
        max_phrases=MAX_PHRASES,
        add_unclassified_row=ADD_UNCLASSIFIED_ROW
    )

    result["ontology_meta"] = meta
    result["ontology_categories_count"] = len(categories)
    result["files"] = {
        "input": str(INPUT_FILE),
        "ontology": str(ONTOLOGY_FILE),
        "outdir": str(OUTDIR)
    }

    outputs = write_outputs(result, OUTDIR)

    print("{} - Fichiers générés".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for k, v in outputs.items():
        print(f"- {k}: {v}")

    print("{} - Total phrases: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), result["total_sentences"]))
    print("{} - Top catégories (5)".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for r in result["stats_rows"][:5]:
        print(f"  {r['category_label']} ({r['category_id']}): {r['occurrences']} -> {r['percent_of_sentences']:.1%}")


if __name__ == "__main__":
    main()

print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))