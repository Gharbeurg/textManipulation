# Bibliothèques
import os
import pandas as pd
from unidecode import unidecode

from datetime import datetime
from colorama import Fore, Back, Style

# fichiers
fichier_entree = "C:/DATA/github/.params/entree.txt"
fichier_sortie = "C:/DATA/github/.data/texte_tage.xlsx"
fichier_mots_cles = "C:/DATA/github/.params/motscles.txt"

#autres variables
df_tags = pd.DataFrame(columns =  ['phrase', 'tags', 'motcle'])
df_tags = df_tags.reset_index(drop=True)
compteur = 0
nbre_phrases_totales = 0
nbre_phrases_traitees = 0
nbre_phrases_erreur = 0

l_phrase = []
l_tags = []
l_toustags = []
l_motcle = []
liste_tags = ""
liste_motcle = ""


# suppression du fichier de sortie
print("{} - Suppression du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
if os.path.exists(fichier_sortie):
    os.remove(fichier_sortie)

# lecture du fichier entree
print("{} - Lecture du fichier d'entree".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
with open(fichier_entree, 'r', encoding="utf-8", errors='ignore') as f:
    for line_texte in f:
        line_texte = unidecode(line_texte) #accents
        line_texte = line_texte.lower() #minuscule

        nbre_phrases_totales +=1
        try:
               
            # lecture du fichier mots clés
            with open(fichier_mots_cles, 'r', encoding="utf-8", errors='ignore') as file_motscles:
                for line in file_motscles:
                    liste_line = line.split(',')
                    for i in range(1,len(liste_line)):
                        if liste_line[i].rstrip() in line_texte and liste_line[0].rstrip() not in liste_tags:
                            liste_tags = liste_tags + liste_line[0] + ';'
                            liste_motcle = liste_motcle + liste_line[i] + ';'
                            l_toustags.append(liste_line[0])

            #suppression du dernier point virgule
            liste_tags = liste_tags[0:len(liste_tags)-1]
            liste_motcle = liste_motcle[0:len(liste_motcle)-1]
            
            #ajout d'une entrée dans le tableau
            l_phrase.append(line_texte.strip())
            l_tags.append(liste_tags)
            l_motcle.append(liste_motcle)
            nbre_phrases_traitees +=1
            liste_tags = ""
            liste_motcle = ""

            print ("[+] {} - phrase traitée OK : {}".format(nbre_phrases_totales, line_texte.strip()))

        except:
            nbre_phrases_erreur += 1
            print(Fore.RED + "[+] {} - Erreur sur la phrase : {}".format(nbre_phrases_totales, line_texte.strip()) + Fore.RESET)

# creation du dataframe
print("{} - Fin du parsing, creation du dataframe".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
df_tags['phrase'] = pd.Series(l_phrase)
df_tags['tags'] = pd.Series(l_tags)
df_tags['motcle'] = pd.Series(l_motcle)

# Comptage du nombre d'occurence par tag
l_tags_ssdoublon = []
i = 0
tag = ""

while i < len(l_toustags):
    if l_toustags[i] not in l_tags_ssdoublon:
        l_tags_ssdoublon.append(l_toustags[i])
    i += 1

for tag in l_tags_ssdoublon:
    count = l_toustags.count(tag)
    print ("{};{}".format(tag, count))

# Comptage
print ("[+] Nombre de phrases totales : {}".format(nbre_phrases_totales))
print ("[+] Nombre de phrases traitées : {}".format(nbre_phrases_traitees))
print ("[+] Nombre de phrases en erreur : {}".format(nbre_phrases_erreur))
    
print("{} - Création du fichier de sortie".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
with open(fichier_sortie, 'w', encoding="utf-8") as fs:
    df_tags.to_excel(fichier_sortie)
    fs.close()
    f.close()

print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))