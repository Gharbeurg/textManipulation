# Bibliothèques
import re, string
from datetime import datetime

# Variables
fichier_entree = "C:/PYTHON/.params/entree.txt"
fichier_sortie = "C:/PYTHON/.data/entree.txt"
lignes_max_par_fichier = 2000000
fileNumber = 1

# Ouverture du gros fichier
print("{} - Lecture du fichier d entree".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
with open(fichier_entree, "rt") as f:
    while True:
        lignes = f.readlines(lignes_max_par_fichier)
        if not lignes:
            break
        file_split = '{}_{}.txt'.format(fichier_sortie, fileNumber)
        print("{} - ecriture fichier {}: ".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), file_split))
        with open(file_split, "wt") as outfile:
            for ligne in lignes:
                outfile.write('%s' % ligne)
        outfile.close()
        fileNumber += 1

# Fin du programme    
print("{} - Fin du programme".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))