#BIBLIOTHEQUES
import requests
from pathlib import Path
from datetime import datetime

# =========================
# PARAMETRES
# =========================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

input_text_file = "C:/PYTHON/.entree/document_extrait.txt"      # fichier TEXTE en entrée
output_file = "C:/PYTHON/.data/synthese_resultat.txt"          # fichier de sortie

model = "gpt-oss:20b"

# Contexte (mémoire de lecture) : plus grand = meilleur sur documents longs, mais plus lent/gourmand
num_ctx = 8192          # 4096 / 8192 / 16384 (si ta machine tient)

# Découpage (par paragraphes + taille max)
chunk_chars = 7000      # 5000-9000 recommandé (plus petit = plus fidèle mais plus d'appels)

# Étape 1 : notes par chunk
chunk_notes_tokens = 320   # 220-380 recommandé (plus grand = notes plus riches)

# Étape 2 : fusion intermédiaire (regroupe N chunks)
group_size = 6             # 4 à 8 recommandé (6 marche bien souvent)
group_merge_tokens = 450   # 350-650 recommandé

# Étape 3 : synthèse finale au format imposé
final_tokens = 1600        # 1200-2200 recommandé

temperature = 0.10         # 0.05-0.15 factuel/stable

debug = True               # True = traces / False = silencieux


# =========================
# FONCTIONS
# =========================

def ts() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def extract_text_from_file(text_path: str) -> str:
    return Path(text_path).read_text(encoding="utf-8", errors="ignore")

def ollama_chat(model: str, prompt: str, num_predict: int, temperature: float, num_ctx: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
            "num_ctx": num_ctx,   # IMPORTANT : augmente la mémoire de contexte
        }
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return ((data.get("message") or {}).get("content")) or ""

def split_paragraphs(text: str):
    # Découpe par paragraphes (meilleur que couper au milieu d’une phrase)
    # On considère qu’un paragraphe est séparé par une ligne vide.
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]

def chunk_by_paragraphs(paragraphs, max_chars: int):
    chunks = []
    buf = []
    size = 0

    for p in paragraphs:
        # +2 pour les sauts de ligne ajoutés lors de la reconstruction
        p_len = len(p) + 2

        # Si un paragraphe est lui-même énorme, on le coupe (cas rare)
        if p_len > max_chars:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            # coupe brutale en sous-morceaux
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end])
                start = end
            continue

        if size + p_len <= max_chars:
            buf.append(p)
            size += p_len
        else:
            chunks.append("\n\n".join(buf))
            buf = [p]
            size = p_len

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks

def group_list(items, group_size: int):
    return [items[i:i+group_size] for i in range(0, len(items), group_size)]

def summarize_text_file(
    input_text_file: str,
    output_file: str,
    model: str,
    num_ctx: int,
    chunk_chars: int,
    chunk_notes_tokens: int,
    group_size: int,
    group_merge_tokens: int,
    final_tokens: int,
    temperature: float,
):
    print(f"{ts()} - Lecture du fichier texte…")
    full_text = extract_text_from_file(input_text_file)

    if debug:
        print(f"{ts()} - Taille texte : {len(full_text)} caractères")
        print(f"{ts()} - Aperçu texte : {repr(full_text[:200])}")

    if len(full_text.strip()) == 0:
        msg = "ERREUR : fichier texte vide."
        Path(output_file).write_text(msg, encoding="utf-8")
        print(msg)
        return ""

    # 1) Découpage propre par paragraphes
    paragraphs = split_paragraphs(full_text)
    chunks = chunk_by_paragraphs(paragraphs, chunk_chars)

    if debug:
        print(f"{ts()} - Paragraphes : {len(paragraphs)}")
        print(f"{ts()} - Chunks : {len(chunks)} (max ~{chunk_chars} chars/chunk)")

    # 2) Notes par chunk (on n’écrit PAS de résumé final ici)
    chunk_notes = []
    for idx, chunk in enumerate(chunks, 1):
        notes_prompt = f"""
Tu lis un extrait d'un document.
Produis des NOTES factuelles (pas un résumé narratif).

Format obligatoire :
- Thèmes abordés : (1 ligne)
- Idées importantes : 6 à 10 puces courtes
- Faits concrets / exemples : 2 à 4 puces (si présents)
- Concepts / mots-clés : 8 à 12 termes

Règles :
- Pas d’invention.
- Ne répète pas l’introduction si l’extrait ne fait pas que de l’introduction.
- Si l’extrait est pauvre : dis "peu d'information dans cet extrait".

EXTRAIT {idx}/{len(chunks)} :
{chunk}
"""
        if debug:
            print(f"{ts()} - Notes chunk {idx}/{len(chunks)}…")

        notes = ollama_chat(
            model=model,
            prompt=notes_prompt,
            num_predict=chunk_notes_tokens,
            temperature=temperature,
            num_ctx=num_ctx
        ).strip()

        if debug:
            print(f"{ts()} - Taille notes chunk {idx} : {len(notes)}")

        chunk_notes.append(f"[CHUNK {idx}] {notes}")

    # 3) Fusion intermédiaire en groupes (évite que la synthèse finale “n’écoute” que le début)
    groups = group_list(chunk_notes, group_size)
    group_summaries = []

    for g_idx, g in enumerate(groups, 1):
        group_prompt = f"""
Tu reçois des notes issues de plusieurs extraits d’un même document.
Fais une consolidation FACTUELLE (pas de style littéraire).

Format obligatoire :
- Thèmes dominants : 3 à 6 puces
- Points importants : 8 à 12 puces
- Nuances / limites mentionnées : 2 à 5 puces

Règles :
- Pas d’invention.
- Évite les répétitions.
- Garde uniquement l’essentiel.

NOTES (groupe {g_idx}/{len(groups)}) :
{chr(10).join(g)}
"""
        if debug:
            print(f"{ts()} - Fusion groupe {g_idx}/{len(groups)}…")

        gsum = ollama_chat(
            model=model,
            prompt=group_prompt,
            num_predict=group_merge_tokens,
            temperature=temperature,
            num_ctx=num_ctx
        ).strip()

        if debug:
            print(f"{ts()} - Taille fusion groupe {g_idx} : {len(gsum)}")

        group_summaries.append(f"[GROUPE {g_idx}] {gsum}")

    # 4) Synthèse finale au FORMAT EXACT voulu
    print(f"{ts()} - Rédaction synthèse finale…")

    final_prompt = f"""
Tu es un assistant de synthèse en français.
Tu reçois des synthèses consolidées de plusieurs parties d’un même document.

Tu dois produire une synthèse avec EXACTEMENT ce format (mêmes titres, même ordre, même style) :

# Synthèse du document
## 1) Résumé
(texte en 2 à 4 paragraphes, sans liste)

## 2) Idées clés du document
(liste de 10 à 14 puces)

Règles STRICTES :
- Respecte exactement les titres et la numérotation : "## 1) Résumé" puis "## 2) Idées clés du document"
- N’ajoute aucune autre section.
- Dans "Résumé" : uniquement du texte (pas de puces, pas de numéros, pas de sous-titres).
- Dans "Idées clés" : uniquement des puces commençant par "- ".
- Évite les répétitions et reste fidèle au contenu.
- Pas d’invention : si une information n’est pas dans les données, ne l’ajoute pas.

DONNÉES À SYNTHÉTISER :
{chr(10).join(group_summaries)}
"""

    final_text = ollama_chat(
        model=model,
        prompt=final_prompt,
        num_predict=final_tokens,
        temperature=temperature,
        num_ctx=num_ctx
    ).strip()

    if debug:
        print(f"{ts()} - Taille synthèse : {len(final_text)} caractères")
        print(f"{ts()} - Aperçu synthèse : {repr(final_text[:200])}")

    if len(final_text.strip()) == 0:
        msg = "ERREUR : synthèse vide."
        Path(output_file).write_text(msg, encoding="utf-8")
        print(msg)
        return ""

    Path(output_file).write_text(final_text, encoding="utf-8")
    print(f"{ts()} - Synthèse enregistrée dans : {output_file}")

    return final_text


if __name__ == "__main__":
    print(f"{ts()} - Lancement")

    summarize_text_file(
        input_text_file=input_text_file,
        output_file=output_file,
        model=model,
        num_ctx=num_ctx,
        chunk_chars=chunk_chars,
        chunk_notes_tokens=chunk_notes_tokens,
        group_size=group_size,
        group_merge_tokens=group_merge_tokens,
        final_tokens=final_tokens,
        temperature=temperature
    )

    print(f"{ts()} - Fin")
