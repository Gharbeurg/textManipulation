"""
Analyse thématique (repérer les sujets) – fichier texte en français.

Ce script :
- lit un .txt
- découpe en paragraphes (ou en blocs si besoin)
- regroupe automatiquement les morceaux par "sujets"
- sort 2 CSV :
  1) topics.csv : liste des sujets + mots-clés
  2) chunks_with_topics.csv : chaque morceau + sujet attribué + score

Dépendances :
  pip install scikit-learn pandas

Lance :
  python analyse_thematique_fr.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# ========= PARAMÈTRES À MODIFIER =========
INPUT_FILE = Path(r"D:/CODING/.params/entree.txt")
OUTPUT_DIR = Path(r"D:/CODING/.data/sorties_thematiques.txt")

NB_TOPICS = 8                 # ex: 5, 8, 12 (plus grand = sujets plus fins)
TOP_WORDS_PER_TOPIC = 12      # nb de mots-clés affichés par sujet

# Découpage :
MIN_CHARS_PER_CHUNK = 200     # ignore les morceaux trop courts
CHUNK_MODE = "paragraphs"     # "paragraphs" (recommandé) ou "blocks"
BLOCK_SIZE_CHARS = 1200       # utilisé seulement si CHUNK_MODE="blocks"
# ========================================
def french_stopwords():
    """
    Liste simple de mots français très fréquents
    (suffisant pour une analyse thématique).
    """
    return [
        "alors","au","aucuns","aussi","autre","avant","avec","avoir","bon",
        "car","ce","cela","ces","ceux","chaque","ci","comme","comment",
        "dans","des","du","dedans","dehors","depuis","devrait","doit",
        "donc","dos","début","elle","elles","en","encore","essai","est",
        "et","eu","fait","faites","fois","font","hors","ici","il","ils",
        "je","juste","la","le","les","leur","là","ma","maintenant","mais",
        "mes","mine","moins","mon","mot","même","ni","nommés","notre",
        "nous","nouveaux","ou","où","par","parce","parole","pas","personnes",
        "peut","peu","pièce","plupart","pour","pourquoi","quand","que",
        "quel","quelle","quelles","quels","qui","sa","sans","ses","seulement",
        "si","sien","son","sont","sous","soyez","sujet","sur","ta","tandis",
        "tellement","tels","tes","ton","tous","tout","trop","très","tu",
        "valeur","voie","voient","vont","votre","vous","vu","ça","étaient",
        "état","étions","été","être"
    ]


def read_text_file(path: Path) -> str:
    """
    Lit un fichier texte.
    Si ton fichier n'est pas en UTF-8, mets encoding="cp1252".
    """
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        # Repli courant Windows
        return path.read_text(encoding="cp1252", errors="replace")


def normalize_text(s: str) -> str:
    # Nettoyage léger pour stabiliser le découpage
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_paragraphs(text: str) -> List[str]:
    # Paragraphes séparés par une ligne vide
    parts = [p.strip() for p in text.split("\n\n")]
    parts = [p for p in parts if len(p) >= MIN_CHARS_PER_CHUNK]
    return parts


def chunk_blocks(text: str, block_size: int) -> List[str]:
    # Blocs de taille fixe si le texte n'a pas de paragraphes propres
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + block_size].strip()
        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            chunks.append(chunk)
        i += block_size
    return chunks


def build_chunks(text: str) -> List[str]:
    if CHUNK_MODE == "paragraphs":
        chunks = chunk_paragraphs(text)
        # Si trop peu de paragraphes, bascule en blocs automatiquement
        if len(chunks) >= 3:
            return chunks
        return chunk_blocks(text, BLOCK_SIZE_CHARS)

    if CHUNK_MODE == "blocks":
        return chunk_blocks(text, BLOCK_SIZE_CHARS)

    raise ValueError("CHUNK_MODE doit être 'paragraphs' ou 'blocks'.")


def extract_topics(chunks: List[str], n_topics: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    On utilise TF-IDF + NMF :
    - TF-IDF : donne plus de poids aux mots qui caractérisent un sujet
    - NMF : regroupe les morceaux en sujets

    (Ce n'est pas parfait, mais très utile pour une première lecture.)
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=french_stopwords(),
        ngram_range=(1, 2),      # 1 mot + groupes de 2 mots ("prise en charge")
        min_df=2,                # ignore les termes vus 1 seule fois
        max_df=0.90              # ignore les termes trop présents partout
    )

    X = vectorizer.fit_transform(chunks)

    model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )

    W = model.fit_transform(X)     # (morceaux x sujets)
    H = model.components_          # (sujets x termes)

    terms = vectorizer.get_feature_names_out()

    # Table "sujets -> mots-clés"
    topics_rows = []
    for topic_id in range(n_topics):
        top_idx = H[topic_id].argsort()[::-1][:TOP_WORDS_PER_TOPIC]
        keywords = [terms[i] for i in top_idx]
        topics_rows.append(
            {"topic_id": topic_id, "keywords": ", ".join(keywords)}
        )
    df_topics = pd.DataFrame(topics_rows)

    # Table "morceaux -> sujet"
    assigned_topic = W.argmax(axis=1)
    confidence = W.max(axis=1)

    df_chunks = pd.DataFrame(
        {
            "chunk_id": list(range(len(chunks))),
            "topic_id": assigned_topic,
            "confidence": confidence,   # plus grand = plus "sûr"
            "text": chunks,
        }
    ).sort_values(["topic_id", "confidence"], ascending=[True, False])

    return df_topics, df_chunks


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_text_file(INPUT_FILE)
    text = normalize_text(raw)
    chunks = build_chunks(text)

    if len(chunks) < 3:
        raise ValueError(
            "Pas assez de contenu exploitable.\n"
            "- Baisse MIN_CHARS_PER_CHUNK (ex: 100)\n"
            "- ou force CHUNK_MODE='blocks'\n"
        )

    df_topics, df_chunks = extract_topics(chunks, NB_TOPICS)

    topics_csv = OUTPUT_DIR / "topics.csv"
    chunks_csv = OUTPUT_DIR / "chunks_with_topics.csv"

    df_topics.to_csv(topics_csv, index=False, encoding="utf-8")
    df_chunks.to_csv(chunks_csv, index=False, encoding="utf-8")

    print("✅ Analyse terminée")
    print(f"- Morceaux analysés : {len(chunks)}")
    print(f"- Sujets trouvés    : {NB_TOPICS}")
    print(f"- Résultats :\n  - {topics_csv}\n  - {chunks_csv}\n")

    print("Aperçu des sujets :")
    for _, row in df_topics.iterrows():
        print(f"  Topic {row['topic_id']}: {row['keywords']}")


if __name__ == "__main__":
    main()
