import os
import pandas as pd

import config
from models.tag_result import TagResult
from utils.console import info, stats


def write(results: list[TagResult], filepath: str) -> None:
    if os.path.exists(filepath):
        os.remove(filepath)

    df = pd.DataFrame({
        "phrase":  [r.phrase                                    for r in results],
        "tags":    [config.SEPARATEUR_TAGS.join(r.tags)         for r in results],
        "motcle":  [config.SEPARATEUR_TAGS.join(r.motscles)     for r in results],
    })

    df.to_excel(filepath, index=False)
    info(f"Fichier de sortie créé : {filepath}")

    _print_tag_counts(results)


def _print_tag_counts(results: list[TagResult]) -> None:
    counts: dict[str, int] = {}
    for r in results:
        for tag in r.tags:
            counts[tag] = counts.get(tag, 0) + 1

    print("\n--- Occurrences par tag ---")
    for tag, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag} ; {count}")
    print()

    total    = len(results)
    traites  = sum(1 for r in results if r.tags)
    sans_tag = total - traites

    stats("Phrases totales",        total)
    stats("Phrases avec tag(s)",    traites)
    stats("Phrases sans tag",       sans_tag)
