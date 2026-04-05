#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)


def extract_video_id(url: str) -> str | None:
    url = url.strip()

    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)

    # youtube.com/watch?v=<id>
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)

    # youtube.com/shorts/<id>
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)

    # youtube.com/embed/<id>
    m = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)

    return None


def fetched_transcript_to_text(fetched_transcript) -> str:
    # fetched_transcript est une liste d’objets "snippet" (petits morceaux)
    # Chaque snippet a un .text
    lines = []
    for snippet in fetched_transcript:
        t = (snippet.text or "").strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def main() -> int:
    # ====== VARIABLES ======
    YOUTUBE_URL = "https://www.youtube.com/watch?v=GJAQCRJZmic"
    OUTPUT_FILE = Path("C:/PYTHON/.data/youtube.txt")
    LANGS = ["fr", "en"]  # ordre de préférence
    # =======================

    video_id = extract_video_id(YOUTUBE_URL)
    if not video_id:
        print("Erreur: impossible d'extraire l'ID vidéo depuis l'URL.")
        return 1

    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=LANGS)
        text = fetched_transcript_to_text(fetched)
    except VideoUnavailable:
        print("Erreur: vidéo indisponible (privée/supprimée/région).")
        return 2
    except TranscriptsDisabled:
        print("Erreur: transcriptions désactivées pour cette vidéo.")
        return 3
    except NoTranscriptFound:
        print("Erreur: aucune transcription trouvée (pas de sous-titres).")
        return 4
    except CouldNotRetrieveTranscript:
        print("Erreur: impossible de récupérer la transcription (blocage / réseau / autre).")
        return 5
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return 6

    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"OK -> {OUTPUT_FILE.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

