import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher

# ============================================================
# 1. PARAMETRES
# ============================================================

FICHIER_ENTREE = r"C:\PYTHON\.entree\roche.xlsx"
FICHIER_SORTIE = r"C:\PYTHON\.data\evenements_roche.xlsx"

COL_DATE = "date"
COL_MOTIF = "motif_lien_interet"
COL_AUTRE_MOTIF = "autre_motif"
COL_INFO_EVENT = "information_evenement"
COL_BENEFICIAIRE = "beneficiaire_nom_prenom_com_name"
COL_MONTANT = "montant"


# ============================================================
# MOTIFS TRANSPARENCE SANTE
# ============================================================

MOTIFS_EVENEMENT_CERTAINS = [
    "frais d'inscription a une manifestation",
    "contrat d'intervenant a une manifestation",
    "contrat de participation a une manifestation",
    "contrat d'achat ou de location d'espaces dans le cadre de manifestations scientifiques",
]

MOTIFS_EVENEMENT_POSSIBLES = [
    "hospitalite - restauration",
    "hospitalite - hebergement",
    "frais de transport",
]

MOTIFS_PARRAINAGE = [
    "parrainage",
]


# ============================================================
# MOTS CLES
# ============================================================

MOTS_EVENEMENT_FORTS = [
    "congres",
    "congress",
    "symposium",
    "conference",
    "colloque",
    "seminaire",
    "webinaire",
    "webinar",
    "workshop",
    "atelier",
    "forum",
    "convention",
    "table ronde",
    "journee scientifique",
    "rencontres",
]

MOTS_EVENEMENT_MOYENS = [
    "reunion",
    "meeting",
    "board",
    "staff",
    "rp",
    "manifestation",
    "evenement",
    "event",
    "journee",
]

MOTS_PARTENARIAT = [
    "partnership agreement",
    "partenariat",
    "partnership",
    "accord de partenariat",
    "collaboration",
]

MOTS_PARRAINAGE_TEXTE = [
    "sponsorship agreement",
    "sponsoring",
    "sponsorship",
    "contrat de parrainage",
    "parrainage",
]

MOTS_PROJET = [
    "programme",
    "program",
    "projet",
    "project",
    "initiative",
    "parcours",
]


# ============================================================
# OUTILS DE NETTOYAGE
# ============================================================

def nettoyer_texte(valeur):
    """Normalise un texte pour faciliter les comparaisons."""

    if pd.isna(valeur):
        return ""

    texte = str(valeur).lower().strip()

    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(
        c for c in texte
        if not unicodedata.combining(c)
    )

    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()


def contient(texte, mots):
    texte = nettoyer_texte(texte)

    for mot in mots:
        mot = nettoyer_texte(mot)

        if re.search(
            r"(?<!\w)" + re.escape(mot) + r"(?!\w)",
            texte
        ):
            return True

    return False


# ============================================================
# LECTURE
# ============================================================

df = pd.read_excel(FICHIER_ENTREE)

print(f"{len(df):,} déclarations chargées")


# ============================================================
# TEXTE CONSOLIDE
# ============================================================

for col in [
    COL_MOTIF,
    COL_AUTRE_MOTIF,
    COL_INFO_EVENT,
    COL_BENEFICIAIRE
]:
    if col not in df.columns:
        df[col] = ""


df["texte_source"] = (
    df[COL_INFO_EVENT].fillna("").astype(str)
    + " | "
    + df[COL_AUTRE_MOTIF].fillna("").astype(str)
)

df["texte_source_normalise"] = (
    df["texte_source"].apply(nettoyer_texte)
)

df["motif_normalise"] = (
    df[COL_MOTIF].apply(nettoyer_texte)
)


# ============================================================
# 1. CLASSIFICATION DE LA NATURE DU LIBELLE
# ============================================================

def classifier_nature(row):

    texte = row["texte_source_normalise"]
    motif = row["motif_normalise"]

    # --------------------------------------------------------
    # Evénement explicite
    # --------------------------------------------------------

    if contient(texte, MOTS_EVENEMENT_FORTS):
        return "EVENEMENT"

    # Motif Transparence Santé explicitement événementiel
    if motif in MOTIFS_EVENEMENT_CERTAINS:
        return "EVENEMENT"

    # --------------------------------------------------------
    # Partenariat
    # --------------------------------------------------------

    if contient(texte, MOTS_PARTENARIAT):
        return "PARTENARIAT"

    # --------------------------------------------------------
    # Parrainage
    # --------------------------------------------------------

    if (
        motif in MOTIFS_PARRAINAGE
        or contient(texte, MOTS_PARRAINAGE_TEXTE)
    ):

        # Si le libellé contient également congrès,
        # symposium, réunion, etc. :
        # on considère qu'un événement est identifiable.

        if (
            contient(texte, MOTS_EVENEMENT_FORTS)
            or contient(texte, MOTS_EVENEMENT_MOYENS)
        ):
            return "EVENEMENT"

        return "PARRAINAGE"

    # --------------------------------------------------------
    # Projet / programme
    # --------------------------------------------------------

    if contient(texte, MOTS_PROJET):
        return "PROJET_PROGRAMME"

    # --------------------------------------------------------
    # Mot événementiel moins discriminant
    # --------------------------------------------------------

    if contient(texte, MOTS_EVENEMENT_MOYENS):
        return "EVENEMENT_POSSIBLE"

    # --------------------------------------------------------
    # Hospitalité seule
    # --------------------------------------------------------

    if motif in MOTIFS_EVENEMENT_POSSIBLES:
        return "DEPENSE_ASSOCIABLE"

    return "INDETERMINE"


df["nature"] = df.apply(
    classifier_nature,
    axis=1
)


# ============================================================
# 2. SCORE DE CONFIANCE
# ============================================================

def calcul_score(row):

    texte = row["texte_source_normalise"]
    motif = row["motif_normalise"]
    nature = row["nature"]

    score = 0

    # --------------------------------------------------------
    # Motif administratif
    # --------------------------------------------------------

    if motif in MOTIFS_EVENEMENT_CERTAINS:
        score += 65

    elif motif in MOTIFS_PARRAINAGE:
        score += 25

    elif motif in MOTIFS_EVENEMENT_POSSIBLES:
        score += 15

    # --------------------------------------------------------
    # Contenu textuel
    # --------------------------------------------------------

    if contient(texte, MOTS_EVENEMENT_FORTS):
        score += 40

    elif contient(texte, MOTS_EVENEMENT_MOYENS):
        score += 25

    # --------------------------------------------------------
    # Information événement explicitement renseignée
    # --------------------------------------------------------

    info = row.get(COL_INFO_EVENT, "")

    if pd.notna(info) and len(str(info).strip()) >= 5:
        score += 15

    # --------------------------------------------------------
    # Réduction si le texte ressemble plus à un accord
    # qu'à un événement
    # --------------------------------------------------------

    if nature == "PARTENARIAT":
        score = min(score, 35)

    elif nature == "PROJET_PROGRAMME":
        score = min(score, 40)

    elif nature == "PARRAINAGE":
        score = min(score, 45)

    return min(score, 100)


df["score_evenement"] = df.apply(
    calcul_score,
    axis=1
)


def niveau_confiance(score):

    if score >= 80:
        return "TRES FORTE"

    if score >= 60:
        return "FORTE"

    if score >= 40:
        return "MOYENNE"

    if score >= 20:
        return "FAIBLE"

    return "TRES FAIBLE"


df["confiance_evenement"] = (
    df["score_evenement"]
    .apply(niveau_confiance)
)


# ============================================================
# 3. EXTRACTION DU TYPE D'EVENEMENT
# ============================================================

def type_evenement(row):

    texte = f" {row['texte_source_normalise']} "

    regles = [
        ("Congrès", ["congres", "congress"]),
        ("Symposium", ["symposium"]),
        ("Conférence", ["conference"]),
        ("Colloque", ["colloque"]),
        ("Séminaire", ["seminaire"]),
        ("Webinaire", ["webinaire", "webinar"]),
        ("Atelier / workshop", ["atelier", "workshop"]),
        ("Forum", ["forum"]),
        ("Convention", ["convention"]),
        ("Table ronde", ["table ronde"]),
        ("Board", ["board"]),
        ("Staff", ["staff"]),
        ("Réunion", ["reunion", "meeting"]),
        ("Relations publiques / RP", [" rp "]),
        ("Journée", ["journee"]),
        ("Rencontre", ["rencontre"]),
    ]

    for categorie, mots in regles:
        if any(
            f" {nettoyer_texte(mot)} " in texte
            for mot in mots
        ):
            return categorie

    if row["nature"] == "EVENEMENT":
        return "Manifestation / événement"

    return ""


df["type_evenement"] = df.apply(
    type_evenement,
    axis=1
)


# ============================================================
# 4. EXTRACTION DU LIBELLE LE PLUS UTILE
# ============================================================

def libelle_principal(row):

    info = row.get(COL_INFO_EVENT, "")
    autre = row.get(COL_AUTRE_MOTIF, "")

    if pd.notna(info):
        info = str(info).strip()

        if len(info) >= 4:
            return info

    if pd.notna(autre):
        autre = str(autre).strip()

        if len(autre) >= 4:
            return autre

    return ""


df["libelle_principal"] = df.apply(
    libelle_principal,
    axis=1
)


# ============================================================
# 5. NETTOYAGE DES REFERENCES INTERNES
# ============================================================

def supprimer_references_internes(texte):

    if pd.isna(texte):
        return ""

    texte = str(texte)

    # Codes ressemblant par exemple à :
    # CCFR25020095
    # GS2604AA0732
    # etc.

    texte = re.sub(
        r"\b[A-Z]{2,6}\d{4,}[A-Z0-9]*\b",
        " ",
        texte,
        flags=re.IGNORECASE
    )

    # mots administratifs peu utiles

    expressions = [
        r"\bsta\b",
        r"\bpartnership agreement\b",
        r"\bsponsorship agreement\b",
        r"\bcontrat de parrainage\b",
        r"\bcontrat de partenariat\b",
    ]

    for expression in expressions:
        texte = re.sub(
            expression,
            " ",
            texte,
            flags=re.IGNORECASE
        )

    texte = re.sub(r"\s+", " ", texte)

    return texte.strip(" -_|")


df["libelle_nettoye"] = (
    df["libelle_principal"]
    .apply(supprimer_references_internes)
)


# ============================================================
# 6. EXTRACTION DES DATES DANS LE TEXTE
# ============================================================

MOIS = (
    "janvier|février|fevrier|mars|avril|mai|juin|"
    "juillet|août|aout|septembre|octobre|novembre|"
    "décembre|decembre"
)


def extraire_date(texte):

    if pd.isna(texte):
        return ""

    texte = str(texte)

    patterns = [
        rf"\b\d{{1,2}}\s+(?:{MOIS})\s+\d{{4}}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
    ]

    for pattern in patterns:

        resultat = re.search(
            pattern,
            texte,
            flags=re.IGNORECASE
        )

        if resultat:
            return resultat.group(0)

    return ""


df["date_evenement_texte"] = (
    df["texte_source"]
    .apply(extraire_date)
)


# ============================================================
# 7. EXTRACTION DE L'ANNEE
# ============================================================

def extraire_annee(texte):

    if pd.isna(texte):
        return ""

    match = re.search(
        r"\b(20\d{2})\b",
        str(texte)
    )

    if match:
        return match.group(1)

    return ""


df["annee_evenement"] = (
    df["texte_source"]
    .apply(extraire_annee)
)


# ============================================================
# 8. NORMALISATION POUR DEDOUBLONNAGE
# ============================================================

def normaliser_nom(texte):

    texte = nettoyer_texte(texte)

    # Supprimer année
    texte = re.sub(
        r"\b20\d{2}\b",
        " ",
        texte
    )

    # Supprimer ponctuation
    texte = re.sub(
        r"[^a-z0-9 ]",
        " ",
        texte
    )

    # Certains mots génériques n'aident pas à reconnaître
    # l'événement
    mots_generiques = [
        "contrat",
        "parrainage",
        "sponsorship",
        "agreement",
        "partnership",
        "manifestation",
    ]

    for mot in mots_generiques:
        texte = re.sub(
            rf"\b{mot}\b",
            " ",
            texte
        )

    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()


df["cle_evenement"] = (
    df["libelle_nettoye"]
    .apply(normaliser_nom)
)


# ============================================================
# 9. CREATION D'UNE REQUETE POUR RECHERCHE INTERNET
# ============================================================

def requete_recherche(row):
    """
    Produit une requête qui pourra ensuite être envoyée
    manuellement ou automatiquement à un moteur de recherche.
    """

    libelle = row["libelle_nettoye"]
    annee = row["annee_evenement"]

    if not libelle:
        return ""

    morceaux = [libelle]

    if annee:
        morceaux.append(annee)

    # Le nom Roche peut aider à trouver parrainage,
    # programme de congrès, communiqué, etc.
    morceaux.append("Roche")

    return " ".join(morceaux)


df["requete_recherche_web"] = df.apply(
    requete_recherche,
    axis=1
)


# ============================================================
# 10. DECISION : FAUT-IL CHERCHER SUR INTERNET ?
# ============================================================

def recherche_web_recommandee(row):

    if row["nature"] in [
        "EVENEMENT",
        "EVENEMENT_POSSIBLE",
        "PARRAINAGE",
        "PARTENARIAT",
        "PROJET_PROGRAMME",
    ]:

        if len(row["libelle_nettoye"]) >= 4:
            return "OUI"

    return "NON"


df["recherche_web"] = df.apply(
    recherche_web_recommandee,
    axis=1
)


# ============================================================
# 11. STATUT FINAL
# ============================================================

def statut_final(row):

    nature = row["nature"]
    score = row["score_evenement"]

    if nature == "EVENEMENT" and score >= 60:
        return "EVENEMENT A IDENTIFIER"

    if nature == "EVENEMENT_POSSIBLE":
        return "A VERIFIER"

    if nature == "PARRAINAGE":
        return "PARRAINAGE - EVENEMENT NON PROUVE"

    if nature == "PARTENARIAT":
        return "PARTENARIAT - EVENEMENT NON PROUVE"

    if nature == "PROJET_PROGRAMME":
        return "PROJET/PROGRAMME - EVENEMENT NON PROUVE"

    if nature == "DEPENSE_ASSOCIABLE":
        return "A RATTACHER A UN EVENEMENT"

    return "NON IDENTIFIE"


df["statut"] = df.apply(
    statut_final,
    axis=1
)


# ============================================================
# 12. TABLE DES LIBELLES UNIQUES
# ============================================================

libelles = (
    df[
        df["libelle_nettoye"].str.len() > 3
    ]
    .groupby(
        [
            "libelle_nettoye",
            "nature",
            "type_evenement",
            "annee_evenement",
            "statut",
            "requete_recherche_web"
        ],
        dropna=False
    )
    .agg(
        nombre_declarations=(
            "libelle_nettoye",
            "size"
        ),

        score_max=(
            "score_evenement",
            "max"
        ),

        premiere_date_declaration=(
            COL_DATE,
            "min"
        ),

        derniere_date_declaration=(
            COL_DATE,
            "max"
        ),

        nombre_beneficiaires=(
            COL_BENEFICIAIRE,
            "nunique"
        ),

        montant_total=(
            COL_MONTANT,
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# 13. LISTE SPECIFIQUE DES EVENEMENTS
# ============================================================

evenements = df[
    df["nature"].isin(
        [
            "EVENEMENT",
            "EVENEMENT_POSSIBLE"
        ]
    )
].copy()


# ============================================================
# 14. LIBELLES A RECHERCHER SUR INTERNET
# ============================================================

a_rechercher = libelles[
    libelles["requete_recherche_web"].str.len() > 0
].copy()

a_rechercher = a_rechercher.sort_values(
    [
        "score_max",
        "nombre_declarations"
    ],
    ascending=[
        False,
        False
    ]
)


# ============================================================
# 15. EXPORT
# ============================================================

with pd.ExcelWriter(
    FICHIER_SORTIE,
    engine="openpyxl"
) as writer:

    # Synthèse principale
    libelles.to_excel(
        writer,
        sheet_name="Libelles_uniques",
        index=False
    )

    # Evénements probables
    evenements.to_excel(
        writer,
        sheet_name="Evenements_probables",
        index=False
    )

    # Ce qu'il faudra rechercher sur Internet
    a_rechercher.to_excel(
        writer,
        sheet_name="Recherche_web",
        index=False
    )

    # Parrainages
    df[
        df["nature"] == "PARRAINAGE"
    ].to_excel(
        writer,
        sheet_name="Parrainages",
        index=False
    )

    # Partenariats
    df[
        df["nature"] == "PARTENARIAT"
    ].to_excel(
        writer,
        sheet_name="Partenariats",
        index=False
    )

    # Projets / programmes
    df[
        df["nature"] == "PROJET_PROGRAMME"
    ].to_excel(
        writer,
        sheet_name="Projets_programmes",
        index=False
    )

    # Hospitalité ou transport à rattacher
    df[
        df["nature"] == "DEPENSE_ASSOCIABLE"
    ].to_excel(
        writer,
        sheet_name="Depenses_a_rattacher",
        index=False
    )

    # Ensemble des données analysées
    df.to_excel(
        writer,
        sheet_name="Toutes_declarations",
        index=False
    )


print()
print("============================================")
print("ANALYSE TERMINEE")
print("============================================")
print(f"Déclarations : {len(df):,}")
print(f"Libellés uniques : {len(libelles):,}")
print(f"Evénements / événements possibles : {len(evenements):,}")
print(f"Libellés à rechercher sur Internet : {len(a_rechercher):,}")
print()
print(f"Fichier généré : {FICHIER_SORTIE}")