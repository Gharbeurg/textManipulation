# -*- coding: utf-8 -*-
"""
Programme d'extraction de lignes et de colonnes depuis un fichier CSV
"transparence" (base médicale), selon deux fichiers de configuration :

  - transparence_colonnes.txt : liste des colonnes à conserver (O) ou non (N)
  - transparence_criteres.txt : critères de filtrage des lignes, avec
    gestion du symbole "%" façon SQL LIKE (contient / commence par /
    se termine par / valeur exacte), plusieurs valeurs possibles par
    colonne (séparées par des virgules, combinées en OU), et combinaison
    en ET entre les différentes colonnes filtrées.

Chemins (codés en dur) :
  - CSV source     : C:\\PYTHON\\.entree\\BasesMedicales\\declarations.csv
  - Colonnes       : C:\\PYTHON\\.entree\\BasesMedicales\\transparence_colonnes.txt
  - Critères       : C:\\PYTHON\\.entree\\BasesMedicales\\transparence_criteres.txt
  - CSV résultat   : C:\\PYTHON\\.data\\resultat_transparence.csv
"""

import csv
import re
import unicodedata


# ---------------------------------------------------------------------------
# Configuration - chemins des fichiers
# ---------------------------------------------------------------------------
FICHIER_CSV_SOURCE = r"C:\PYTHON\.entree\BasesMedicales\declarations.csv"
FICHIER_COLONNES = r"C:\PYTHON\.entree\BasesMedicales\transparence_colonnes.txt"
FICHIER_CRITERES = r"C:\PYTHON\.entree\BasesMedicales\transparence_criteres.txt"
FICHIER_RESULTAT = r"C:\PYTHON\.data\resultat_transparence.csv"

SEPARATEUR_CSV = ";"
ENCODAGE = "utf-8"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def normaliser(texte):
    """
    Normalise une chaîne pour une comparaison insensible à la casse
    et aux accents : "Élève" et "eleve" deviennent identiques.
    """
    if texte is None:
        return ""
    texte = str(texte)
    # Décompose les caractères accentués (é -> e + accent) puis retire les accents
    texte_decompose = unicodedata.normalize("NFKD", texte)
    texte_sans_accents = "".join(
        c for c in texte_decompose if not unicodedata.combining(c)
    )
    return texte_sans_accents.lower()


def lire_colonnes_a_garder(chemin_fichier):
    """
    Lit le fichier transparence_colonnes.txt (format nom_colonne=O ou N)
    et retourne la liste ordonnée des colonnes à conserver (valeur "O").
    """
    colonnes_a_garder = []
    with open(chemin_fichier, encoding=ENCODAGE) as f:
        for numero_ligne, ligne in enumerate(f, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            if "=" not in ligne:
                print(f"  ATTENTION : ligne {numero_ligne} ignorée (format invalide) : {ligne}")
                continue
            nom_colonne, valeur = ligne.rsplit("=", 1)
            nom_colonne = nom_colonne.strip()
            valeur = valeur.strip().upper()
            if valeur == "O":
                colonnes_a_garder.append(nom_colonne)
    return colonnes_a_garder


def lire_criteres(chemin_fichier):
    """
    Lit le fichier transparence_criteres.txt (format nom_colonne;"valeur")
    et retourne un dictionnaire {nom_colonne: [motif1, motif2, ...]}
    ne contenant que les colonnes ayant au moins une valeur de filtre non vide.

    Plusieurs valeurs pour une même colonne sont séparées par une virgule
    à l'intérieur des guillemets, ex : raison_sociale;"%ROCHE%,%NOVARTIS%"
    """
    criteres = {}
    with open(chemin_fichier, encoding=ENCODAGE) as f:
        for numero_ligne, ligne in enumerate(f, start=1):
            ligne = ligne.rstrip("\r\n")
            if not ligne.strip():
                continue
            if ";" not in ligne:
                print(f"  ATTENTION : ligne {numero_ligne} ignorée (format invalide) : {ligne}")
                continue
            nom_colonne, valeur_brute = ligne.split(";", 1)
            nom_colonne = nom_colonne.strip()
            valeur_brute = valeur_brute.strip()

            # Retire les guillemets englobants s'ils existent
            if valeur_brute.startswith('"') and valeur_brute.endswith('"'):
                valeur_brute = valeur_brute[1:-1]

            if not valeur_brute:
                continue  # pas de critère pour cette colonne

            # Plusieurs valeurs séparées par une virgule
            motifs = [m.strip() for m in valeur_brute.split(",") if m.strip()]
            if motifs:
                criteres[nom_colonne] = motifs
    return criteres


def construire_regex_like(motif):
    """
    Convertit un motif façon SQL LIKE (avec % en début/fin/absent) en
    fonction de test appliquée sur une valeur déjà normalisée.

      "%valeur%" -> contient "valeur"
      "%valeur"  -> se termine par "valeur"
      "valeur%"  -> commence par "valeur"
      "valeur"   -> égal exactement à "valeur"
    """
    commence_par_pourcent = motif.startswith("%")
    finit_par_pourcent = motif.endswith("%")

    coeur = motif
    if commence_par_pourcent:
        coeur = coeur[1:]
    if finit_par_pourcent:
        coeur = coeur[:-1]

    coeur_normalise = normaliser(coeur)
    motif_regex = re.escape(coeur_normalise)

    if commence_par_pourcent and finit_par_pourcent:
        motif_regex = motif_regex  # contient, pas d'ancrage
    elif finit_par_pourcent and not commence_par_pourcent:
        motif_regex = "^" + motif_regex  # commence par
    elif commence_par_pourcent and not finit_par_pourcent:
        motif_regex = motif_regex + "$"  # se termine par
    else:
        motif_regex = "^" + motif_regex + "$"  # valeur exacte

    return re.compile(motif_regex)


def ligne_correspond_aux_criteres(ligne_dict, criteres_compiles):
    """
    Vérifie qu'une ligne du CSV (dict colonne -> valeur) respecte tous les
    critères (ET entre colonnes, OU entre les valeurs d'une même colonne).
    Une valeur vide dans la ligne ne peut jamais correspondre à un critère.
    """
    for nom_colonne, liste_regex in criteres_compiles.items():
        valeur_ligne = ligne_dict.get(nom_colonne, "")
        if not valeur_ligne or not valeur_ligne.strip():
            return False  # valeur vide -> ne correspond jamais à un critère

        valeur_normalisee = normaliser(valeur_ligne)

        # OU entre les différentes valeurs possibles du même critère
        if not any(regex.search(valeur_normalisee) for regex in liste_regex):
            return False  # aucune des valeurs ne correspond -> ligne rejetée

    return True  # tous les critères (ET) sont satisfaits


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Extraction transparence - démarrage")
    print("=" * 70)

    # 1. Lecture des colonnes à conserver
    print(f"\nLecture des colonnes à conserver : {FICHIER_COLONNES}")
    colonnes_a_garder = lire_colonnes_a_garder(FICHIER_COLONNES)
    print(f"  -> {len(colonnes_a_garder)} colonne(s) à conserver : {colonnes_a_garder}")

    # 2. Lecture des critères de filtrage
    print(f"\nLecture des critères de filtrage : {FICHIER_CRITERES}")
    criteres_bruts = lire_criteres(FICHIER_CRITERES)
    criteres_compiles = {
        colonne: [construire_regex_like(motif) for motif in motifs]
        for colonne, motifs in criteres_bruts.items()
    }
    for colonne, motifs in criteres_bruts.items():
        print(f"  -> critère sur '{colonne}' : {motifs}")
    if not criteres_bruts:
        print("  -> aucun critère de filtrage trouvé (toutes les lignes seront conservées)")

    # 3. Lecture du CSV source + filtrage des lignes + sélection des colonnes
    print(f"\nLecture du fichier source : {FICHIER_CSV_SOURCE}")
    lignes_resultat = []
    nb_lignes_lues = 0

    with open(FICHIER_CSV_SOURCE, encoding=ENCODAGE, newline="") as f_source:
        lecteur = csv.DictReader(f_source, delimiter=SEPARATEUR_CSV)

        # Vérification que les colonnes demandées existent bien dans le CSV source
        colonnes_manquantes = [c for c in colonnes_a_garder if c not in lecteur.fieldnames]
        if colonnes_manquantes:
            print(f"  ATTENTION : colonnes absentes du CSV source et ignorées : {colonnes_manquantes}")
            colonnes_a_garder = [c for c in colonnes_a_garder if c not in colonnes_manquantes]

        for ligne in lecteur:
            nb_lignes_lues += 1
            if ligne_correspond_aux_criteres(ligne, criteres_compiles):
                ligne_filtree = {colonne: ligne.get(colonne, "") for colonne in colonnes_a_garder}
                lignes_resultat.append(ligne_filtree)

    # 4. Écriture du fichier résultat
    print(f"\nÉcriture du fichier résultat : {FICHIER_RESULTAT}")
    with open(FICHIER_RESULTAT, "w", encoding=ENCODAGE, newline="") as f_resultat:
        ecrivain = csv.DictWriter(
            f_resultat,
            fieldnames=colonnes_a_garder,
            delimiter=SEPARATEUR_CSV,
        )
        ecrivain.writeheader()
        ecrivain.writerows(lignes_resultat)

    # 5. Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Lignes lues dans le fichier source     : {nb_lignes_lues}")
    print(f"Lignes conservées dans le résultat      : {len(lignes_resultat)}")
    print(f"Colonnes conservées dans le résultat    : {len(colonnes_a_garder)}")
    print(f"Fichier résultat écrit avec succès      : {FICHIER_RESULTAT}")
    print("=" * 70)


if __name__ == "__main__":
    main()