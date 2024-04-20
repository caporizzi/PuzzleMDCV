# In :
#   - tabIn  : tableau des couleurs d'un coté de piece
#   - a      : nombre de mesures a prendre dans le tableau
# Out : 
#   tableau des couleurs aux points de mesure
def cutColorBand(tabIn,a):
    x = len(tabIn) # taill du tableau

    xMid = int(x / 2) # index central du tableau
    xStep = int(x / a) # taille des etapes

    # la methode utilisée pose le/les premiers points au centre afin de rendre la mesure "symetrique" (quel que soit le sens dans lequel on passe le tableau dans la fonction,
    # on devrait avoir le meme resultat).

    tabX = []
    if a % 2 == 0: 
        # si le nombre d'etapes est pair 
        tabX = [xMid - int(xStep / 2), xMid + int(xStep / 2)] # deux etapes a centrer, on fait un demi-pas dans chaque direction depuis le centre
    else:
        # sinon
        tabX = [xMid] # premiere etape au centre

    while(len(tabX) < a):
        # tant que le tableau n'a pas le bon nombre de mesures
        tabX = [tabX[0] - xStep] + tabX + [tabX[-1] + xStep] # append les deux etapes de chaque coté du tableau

    # verifie que le tableau de sortie a le non nombre de mesures
    assert(len(tabX) == a, "y a pas le bon nombre d'etapes chef.")

    # utilise le tableau d'indexs qu'on a créé au dessus pour aller chercher les valeurs correspondantes dans le tableau d'entrée
    tabOut = []
    for i in tabX:
        tabOut.append(tabIn[i])

    #retourne le tableau des couleurs echantillonnées
    return tabOut