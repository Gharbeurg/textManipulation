
"""
Optimisation des images d'un PowerPoint (.pptx).

Fonctionnalités :
- Conversion PNG opaque vers JPEG
- Compression JPEG et PNG
- Protection des images transparentes
- Redimensionnement facultatif
- Conservation de la structure PowerPoint
- Rapport détaillé CSV

Sécurité :
- La présentation source n'est jamais modifiée.
- Les images contenant au moins un pixel transparent,
  même partiellement, restent intactes.
- Les références internes sont mises à jour si une
  image PNG devient JPEG.
- Les présentations signées numériquement sont refusées.
"""

from __future__ import annotations

import csv
import io
import os
import posixpath
import tempfile
import xml.etree.ElementTree as ET

from collections import Counter
from copy import copy
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. VARIABLES DE CONFIGURATION
# ============================================================

FICHIER_SOURCE = Path(r"C:\PYTHON\.entree\SourcesPowerpoint\chiesi.pptx")
FICHIER_SORTIE = Path(r"C:\PYTHON\.entree\SourcesPowerpoint\chiesi_optimisee.pptx")
FICHIER_RAPPORT = Path(r"C:\PYTHON\.entree\SourcesPowerpoint\rapport_optimisation_images.csv")

# Types d'optimisation
CONVERTIR_PNG_OPAQUES_EN_JPEG = True
RECOMPRESSER_JPEG = True
OPTIMISER_PNG_OPAQUES = True

# Protection de la transparence
CONSERVER_PNG_TRANSPARENTS_INTACTS = True

# Qualité des images
QUALITE_JPEG = 80
GAIN_MINIMUM_POURCENT = 20

# Protection des schémas et icônes
PROTEGER_IMAGES_PEUX_COULEURS = True
SEUIL_NOMBRE_COULEURS = 256

# Redimensionnement facultatif
REDIMENSIONNER = False
LARGEUR_MAX = 2400
HAUTEUR_MAX = 1600

# Fichiers et affichage
AUTORISER_ECRASEMENT_SORTIE = False
GENERER_RAPPORT_CSV = True
AFFICHER_DETAIL_IMAGES = True


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

NAMESPACE_REL = (
    "http://schemas.openxmlformats.org/"
    "package/2006/relationships"
)

NAMESPACE_CT = (
    "http://schemas.openxmlformats.org/"
    "package/2006/content-types"
)


def taille_lisible(nombre_octets):
    taille = float(nombre_octets)

    for unite in ("o", "Ko", "Mo", "Go"):
        if taille < 1024 or unite == "Go":
            return f"{taille:.1f} {unite}"
        taille /= 1024


def correspond_a_image(nom):
    return (
        nom.startswith("ppt/media/")
        and nom.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    )


def est_transparente(image):
    """
    Détecte les images contenant une transparence,
    même partielle.
    """

    if (
        image.mode not in ("P", "LA", "RGBA")
        and "transparency" not in image.info
    ):
        return False

    alpha = image.convert("RGBA").getchannel("A")

    return alpha.getextrema()[0] < 255


def est_image_peu_coloree(image):
    """
    Identifie les images avec peu de couleurs
    afin de protéger certains schémas et icônes.
    """

    if not PROTEGER_IMAGES_PEUX_COULEURS:
        return False

    miniature = image.convert("RGB")

    miniature.thumbnail(
        (256, 256),
        Image.Resampling.LANCZOS
    )

    couleurs = miniature.getcolors(
        maxcolors=SEUIL_NOMBRE_COULEURS
    )

    return couleurs is not None


def redimensionner_si_necessaire(image):

    if not REDIMENSIONNER:
        return image, False

    if (
        image.width <= LARGEUR_MAX
        and image.height <= HAUTEUR_MAX
    ):
        return image, False

    dimensions = image.size
    copie = image.copy()

    copie.thumbnail(
        (LARGEUR_MAX, HAUTEUR_MAX),
        Image.Resampling.LANCZOS
    )

    return copie, copie.size != dimensions


# ============================================================
# 3. COMPRESSION DES IMAGES
# ============================================================

def enregistrer_jpeg(image, info_originale=None):

    original = info_originale or {}

    image_rgb = image.convert("RGB")

    fichier = io.BytesIO()

    options = {
        "format": "JPEG",
        "quality": QUALITE_JPEG,
        "optimize": True,
        "progressive": True,
    }

    for cle in ("icc_profile", "dpi", "exif"):
        valeur = original.get(cle)

        if valeur:
            options[cle] = valeur

    image_rgb.save(fichier, **options)

    return fichier.getvalue()


def enregistrer_png_sans_perte(
    image,
    info_originale=None
):
    """
    Optimise un PNG sans modification des pixels.
    """

    original = info_originale or {}

    fichier = io.BytesIO()

    options = {
        "format": "PNG",
        "optimize": True,
        "compress_level": 9
    }

    for cle in ("icc_profile", "dpi"):
        valeur = original.get(cle)

        if valeur:
            options[cle] = valeur

    image.save(fichier, **options)

    resultat = fichier.getvalue()

    # Vérification : les pixels doivent rester identiques.

    with Image.open(io.BytesIO(resultat)) as controle:

        controle.load()

        if (
            controle.size != image.size
            or controle.convert("RGBA").tobytes()
            != image.convert("RGBA").tobytes()
        ):
            raise ValueError(
                "La recompression PNG a modifié les pixels."
            )

    return resultat


def est_gain_suffisant(
    taille_originale,
    taille_nouvelle
):

    seuil = (
        1 - GAIN_MINIMUM_POURCENT / 100
    )

    return taille_nouvelle < taille_originale * seuil


# ============================================================
# 4. OPTIMISATION D'UNE IMAGE
# ============================================================

def optimiser_une_image(
    nom,
    donnees,
    autoriser_conversion
):
    """
    Analyse et optimise une image.

    Retourne :
    - Les octets optimisés
    - La nouvelle extension éventuelle
    - La description du traitement
    """

    extension = Path(nom).suffix.lower()

    try:

        with Image.open(io.BytesIO(donnees)) as image:

            image.load()

            info_originale = dict(image.info)

            # --------------------------------------------
            # Protection absolue de la transparence
            # --------------------------------------------

            if est_transparente(image):

                return (
                    donnees,
                    None,
                    "Transparence conservée : image intacte"
                )

            # --------------------------------------------
            # Redimensionnement éventuel
            # --------------------------------------------

            orientation_exif = image.getexif().get(
                274, 1
            )

            if (
                extension in (".jpg", ".jpeg")
                and orientation_exif != 1
            ):
                version_reduite = image
                reduction = False

            else:
                version_reduite, reduction = (
                    redimensionner_si_necessaire(image)
                )

            candidats = []

            # --------------------------------------------
            # Traitement des PNG opaques
            # --------------------------------------------

            if extension == ".png":

                if OPTIMISER_PNG_OPAQUES:

                    png_optimise = (
                        enregistrer_png_sans_perte(
                            version_reduite,
                            info_originale
                        )
                    )

                    description = "PNG optimisé"

                    if reduction:
                        description += " et redimensionné"

                    candidats.append((
                        png_optimise,
                        None,
                        description
                    ))

                if (
                    CONVERTIR_PNG_OPAQUES_EN_JPEG
                    and autoriser_conversion
                    and not est_image_peu_coloree(
                        version_reduite
                    )
                ):

                    jpeg = enregistrer_jpeg(
                        version_reduite,
                        info_originale
                    )

                    description = "PNG converti en JPEG"

                    if reduction:
                        description += " et redimensionné"

                    candidats.append((
                        jpeg,
                        ".jpg",
                        description
                    ))

            # --------------------------------------------
            # Traitement des JPEG
            # --------------------------------------------

            elif extension in (".jpg", ".jpeg"):

                if RECOMPRESSER_JPEG or reduction:

                    jpeg = enregistrer_jpeg(
                        version_reduite,
                        info_originale
                    )

                    description = "JPEG recompressé"

                    if reduction:
                        description += " et redimensionné"

                    candidats.append((
                        jpeg,
                        None,
                        description
                    ))

            # --------------------------------------------
            # Sélection du meilleur résultat
            # --------------------------------------------

            candidats = [
                candidat
                for candidat in candidats
                if est_gain_suffisant(
                    len(donnees),
                    len(candidat[0])
                )
            ]

            if not candidats:

                return (
                    donnees,
                    None,
                    "Image conservée "
                    "(gain insuffisant ou protégée)"
                )

            gagnant = min(
                candidats,
                key=lambda candidat: len(candidat[0])
            )

            return gagnant

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as erreur:

        return (
            donnees,
            None,
            f"Image non modifiée : {erreur}"
        )


# ============================================================
# 5. GESTION DES RÉFÉRENCES POWERPOINT
# ============================================================

def origine_d_un_fichier_rels(nom_rels):
    """
    Retrouve le fichier source d'un fichier de relations.
    """

    dossier, nom = posixpath.split(nom_rels)

    if (
        posixpath.basename(dossier) != "_rels"
        or not nom.endswith(".rels")
    ):
        return None

    dossier_origine = posixpath.dirname(dossier)

    return posixpath.join(
        dossier_origine,
        nom[:-5]
    )


def chemin_cible(nom_rels, cible):
    """
    Retrouve le fichier ciblé par une relation interne.
    """

    elements = urlsplit(cible)

    if (
        elements.scheme
        or elements.netloc
        or not elements.path
    ):
        return None

    chemin = unquote(elements.path)

    if chemin.startswith("/"):

        resultat = posixpath.normpath(
            chemin.lstrip("/")
        )

    else:

        origine = origine_d_un_fichier_rels(
            nom_rels
        )

        if origine is None:
            return None

        resultat = posixpath.normpath(
            posixpath.join(
                posixpath.dirname(origine),
                chemin
            )
        )

    if (
        resultat == ".."
        or resultat.startswith("../")
    ):
        return None

    return resultat


def remplacer_cible(
    nom_rels,
    ancienne_cible,
    nouveau_nom
):
    """
    Modifie la référence d'une image convertie.
    """

    elements = urlsplit(ancienne_cible)

    origine = origine_d_un_fichier_rels(
        nom_rels
    )

    if elements.path.startswith("/"):

        chemin = "/" + nouveau_nom

    else:

        chemin = posixpath.relpath(
            nouveau_nom,
            posixpath.dirname(origine) or "."
        )

    nouveau_chemin = quote(
        chemin,
        safe="/._-~"
    )

    return urlunsplit((
        elements.scheme,
        elements.netloc,
        nouveau_chemin,
        elements.query,
        elements.fragment
    ))


def xml_vers_octets(
    racine,
    namespace_defaut
):

    ET.register_namespace(
        "",
        namespace_defaut
    )

    return ET.tostring(
        racine,
        encoding="utf-8",
        xml_declaration=True
    )


def creer_info_renomme(
    original,
    nouveau_nom
):
    """
    Copie les informations ZIP d'une image renommée.
    """

    copie = copy(original)
    copie.filename = nouveau_nom

    return copie


# ============================================================
# 6. VÉRIFICATIONS ET RAPPORT
# ============================================================

def verifier_entree_sortie():

    source = Path(
        FICHIER_SOURCE
    ).expanduser().resolve()

    destination = Path(
        FICHIER_SORTIE
    ).expanduser().resolve()

    if not source.is_file():

        raise FileNotFoundError(
            f"PowerPoint introuvable : {source}"
        )

    if (
        source.suffix.lower() != ".pptx"
        or destination.suffix.lower() != ".pptx"
    ):

        raise ValueError(
            "Seuls les fichiers .pptx sont acceptés."
        )

    if source == destination:

        raise ValueError(
            "Le fichier source et le fichier de sortie "
            "doivent être différents."
        )

    if (
        destination.exists()
        and not AUTORISER_ECRASEMENT_SORTIE
    ):

        raise FileExistsError(
            f"Le fichier de sortie existe déjà : "
            f"{destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CONSERVER_PNG_TRANSPARENTS_INTACTS:

        raise ValueError(
            "La protection des images transparentes "
            "ne peut pas être désactivée."
        )

    if not 1 <= QUALITE_JPEG <= 95:

        raise ValueError(
            "QUALITE_JPEG doit être compris entre 1 et 95."
        )

    if not 0 <= GAIN_MINIMUM_POURCENT < 100:

        raise ValueError(
            "GAIN_MINIMUM_POURCENT doit être "
            "compris entre 0 et 100."
        )

    if min(LARGEUR_MAX, HAUTEUR_MAX) < 1:

        raise ValueError(
            "Les dimensions maximales doivent "
            "être positives."
        )

    return source, destination


def sauvegarder_rapport(lignes):

    if not GENERER_RAPPORT_CSV:
        return

    chemin = Path(
        FICHIER_RAPPORT
    ).expanduser()

    if chemin.parent != Path("."):

        chemin.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    colonnes = (
        "image_originale",
        "image_finale",
        "poids_avant_octets",
        "poids_apres_octets",
        "gain_pourcent",
        "traitement"
    )

    with chemin.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        redacteur = csv.DictWriter(
            fichier,
            fieldnames=colonnes,
            delimiter=";"
        )

        redacteur.writeheader()
        redacteur.writerows(lignes)

    print(
        f"Rapport CSV : {chemin.resolve()}"
    )


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    source, destination = (
        verifier_entree_sortie()
    )

    print(f"Fichier source : {source}")
    print("Analyse et optimisation des images...")

    with ZipFile(source, "r") as archive:

        if archive.testzip() is not None:

            raise BadZipFile(
                "Le PowerPoint source est corrompu."
            )

        noms = archive.namelist()

        if len(noms) != len(set(noms)):

            raise ValueError(
                "Le PowerPoint contient des "
                "fichiers ZIP dupliqués."
            )

        ensemble_noms = set(noms)

        if "[Content_Types].xml" not in ensemble_noms:

            raise ValueError(
                "Ce fichier n'est pas un PowerPoint "
                "OOXML valide."
            )

        if any(
            "_xmlsignatures/" in nom.lower()
            for nom in noms
        ):

            raise ValueError(
                "Présentation signée numériquement : "
                "modification refusée."
            )

        # ------------------------------------------------
        # Identifier les images référencées
        # ------------------------------------------------

        xml_rels = {}
        images_referencees = Counter()

        for nom in noms:

            if not nom.endswith(".rels"):
                continue

            racine = ET.fromstring(
                archive.read(nom)
            )

            xml_rels[nom] = racine

            for relation in racine:

                if relation.attrib.get(
                    "TargetMode", "Internal"
                ) == "External":
                    continue

                cible = chemin_cible(
                    nom,
                    relation.attrib.get(
                        "Target", ""
                    )
                )

                if cible in ensemble_noms:
                    images_referencees[cible] += 1

        # ------------------------------------------------
        # Optimiser les images
        # ------------------------------------------------

        remplacements = {}
        renommages = {}
        noms_reserves = set(noms)
        rapport = []

        for nom in noms:

            if not correspond_a_image(nom):
                continue

            originales = archive.read(nom)

            utilisee = (
                images_referencees[nom] > 0
            )

            donnees, extension, description = (
                optimiser_une_image(
                    nom,
                    originales,
                    utilisee
                )
            )

            nouveau_nom = nom

            # Conversion éventuelle PNG -> JPEG

            if (
                extension is not None
                and len(donnees) < len(originales)
            ):

                candidat_base = (
                    posixpath.splitext(nom)[0]
                    + extension
                )

                candidat = candidat_base
                compteur = 2

                while candidat in noms_reserves:

                    candidat = (
                        posixpath.splitext(
                            candidat_base
                        )[0]
                        + f"_{compteur}"
                        + extension
                    )

                    compteur += 1

                nouveau_nom = candidat

                noms_reserves.add(
                    nouveau_nom
                )

                renommages[nom] = nouveau_nom

            if donnees != originales:

                remplacements[nom] = donnees

            gain = (
                100 * (
                    1 - len(donnees) / len(originales)
                )
                if originales else 0
            )

            rapport.append({
                "image_originale": nom,
                "image_finale": nouveau_nom,
                "poids_avant_octets": len(originales),
                "poids_apres_octets": len(donnees),
                "gain_pourcent": f"{gain:.1f}",
                "traitement": description,
            })

            if AFFICHER_DETAIL_IMAGES:

                print(
                    f"  {nom}: {description} | "
                    f"{taille_lisible(len(originales))} "
                    f"-> {taille_lisible(len(donnees))} "
                    f"({gain:.1f} %)"
                )

        # ------------------------------------------------
        # Mettre à jour les références des images
        # ------------------------------------------------

        nouvelles_relations = {}

        for nom, racine in xml_rels.items():

            modifie = False

            for relation in racine:

                if relation.attrib.get(
                    "TargetMode", "Internal"
                ) == "External":
                    continue

                ancien = relation.attrib.get(
                    "Target", ""
                )

                cible = chemin_cible(
                    nom,
                    ancien
                )

                if cible in renommages:

                    relation.set(
                        "Target",
                        remplacer_cible(
                            nom,
                            ancien,
                            renommages[cible]
                        )
                    )

                    modifie = True

            if modifie:

                nouvelles_relations[nom] = (
                    xml_vers_octets(
                        racine,
                        NAMESPACE_REL
                    )
                )

        # ------------------------------------------------
        # Mettre à jour les formats du PowerPoint
        # ------------------------------------------------

        nouvelles_declarations = None

        if renommages:

            racine = ET.fromstring(
                archive.read(
                    "[Content_Types].xml"
                )
            )

            for elem in racine.findall(
                f"{{{NAMESPACE_CT}}}Override"
            ):

                partie = elem.attrib.get(
                    "PartName", ""
                ).lstrip("/")

                if partie in renommages:

                    elem.set(
                        "PartName",
                        "/" + renommages[partie]
                    )

                    elem.set(
                        "ContentType",
                        "image/jpeg"
                    )

            types_defaut = {
                e.attrib.get(
                    "Extension", ""
                ).lower()

                for e in racine.findall(
                    f"{{{NAMESPACE_CT}}}Default"
                )
            }

            if "jpg" not in types_defaut:

                ET.SubElement(
                    racine,
                    f"{{{NAMESPACE_CT}}}Default",
                    {
                        "Extension": "jpg",
                        "ContentType": "image/jpeg"
                    }
                )

            nouvelles_declarations = (
                xml_vers_octets(
                    racine,
                    NAMESPACE_CT
                )
            )

        # ------------------------------------------------
        # Reconstruire le PowerPoint
        # ------------------------------------------------

        descripteur, nom_temporaire = (
            tempfile.mkstemp(
                prefix="pptx_optimisation_",
                suffix=".tmp",
                dir=str(destination.parent)
            )
        )

        os.close(descripteur)

        try:

            with ZipFile(
                nom_temporaire,
                "w",
                allowZip64=True
            ) as sortie:

                sortie.comment = archive.comment

                for info in archive.infolist():

                    nom = info.filename

                    if nom in nouvelles_relations:

                        contenu = (
                            nouvelles_relations[nom]
                        )

                    elif (
                        nom == "[Content_Types].xml"
                        and nouvelles_declarations
                        is not None
                    ):

                        contenu = nouvelles_declarations

                    elif nom in remplacements:

                        contenu = remplacements[nom]

                    else:

                        contenu = archive.read(nom)

                    if nom in renommages:

                        nouvelle_info = (
                            creer_info_renomme(
                                info,
                                renommages[nom]
                            )
                        )

                        sortie.writestr(
                            nouvelle_info,
                            contenu
                        )

                    else:

                        sortie.writestr(
                            info,
                            contenu
                        )

            # --------------------------------------------
            # Vérification du fichier final
            # --------------------------------------------

            with ZipFile(
                nom_temporaire,
                "r"
            ) as controle:

                erreur = controle.testzip()

                if erreur is not None:

                    raise BadZipFile(
                        f"Fichier corrompu : {erreur}"
                    )

                noms_finaux = set(
                    controle.namelist()
                )

                for ancien, nouveau in (
                    renommages.items()
                ):

                    if (
                        ancien in noms_finaux
                        or nouveau not in noms_finaux
                    ):

                        raise ValueError(
                            "Renommage non conforme : "
                            f"{ancien} -> {nouveau}"
                        )

                for nom in nouvelles_relations:

                    racine = ET.fromstring(
                        controle.read(nom)
                    )

                    for relation in racine:

                        if relation.attrib.get(
                            "TargetMode", "Internal"
                        ) == "External":
                            continue

                        cible = chemin_cible(
                            nom,
                            relation.attrib.get(
                                "Target", ""
                            )
                        )

                        if (
                            cible in renommages
                            or (
                                cible is not None
                                and cible.startswith(
                                    "ppt/media/"
                                )
                                and cible not in noms_finaux
                            )
                        ):

                            raise ValueError(
                                "Lien vers une image "
                                f"manquante : {cible}"
                            )

            if (
                destination.exists()
                and not AUTORISER_ECRASEMENT_SORTIE
            ):

                raise FileExistsError(
                    "Le fichier de sortie "
                    "existe désormais."
                )

            os.replace(
                nom_temporaire,
                destination
            )

        finally:

            if os.path.exists(nom_temporaire):

                os.remove(
                    nom_temporaire
                )

    # ====================================================
    # 8. BILAN
    # ====================================================

    poids_avant = source.stat().st_size
    poids_apres = destination.stat().st_size

    economie = (
        100 * (
            1 - poids_apres / poids_avant
        )
        if poids_avant else 0
    )

    compteur = sum(
        1
        for ligne in rapport
        if (
            ligne["poids_apres_octets"]
            < ligne["poids_avant_octets"]
        )
    )

    print("\n" + "=" * 60)

    print(
        f"Images analysées      : {len(rapport)}"
    )

    print(
        f"Images allégées       : {compteur}"
    )

    print(
        f"Conversions PNG/JPEG  : {len(renommages)}"
    )

    print(
        f"PowerPoint original   : "
        f"{taille_lisible(poids_avant)}"
    )

    print(
        f"PowerPoint optimisé   : "
        f"{taille_lisible(poids_apres)}"
    )

    print(
        f"Réduction totale      : {economie:.1f} %"
    )

    print(
        f"Fichier créé          : {destination}"
    )

    print("=" * 60)

    sauvegarder_rapport(rapport)


# ============================================================
# 9. EXÉCUTION
# ============================================================

if __name__ == "__main__":
    main()
