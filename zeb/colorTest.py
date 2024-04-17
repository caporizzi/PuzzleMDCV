# In :
#   - col1: premiere couleur a comparer, format tableau RGB [a,b,c] avec a,b,c des nombres entre 0 et 255
#   - col2: seconde couleur a comparer, format tableau RGB [a,b,c] avec a,b,c des nombres entre 0 et 255
# Out :
#   renvoie un "taux de proximité" compris entre 0 et 1
def colorTest(col1, col2):
    assert(len(col1) == len(col2), "les deux tableaux font pas la meme taille chef.")

    tot = 0
    for i in range(0,len(col1)):
        tot += (abs(col1[i][0] - col2[i][0]) + abs(col1[i][1] - col2[i][1]) + abs(col1[i][2] - col2[i][2])) / 765
    
    return 1 - (tot / len(col1))
    


#tests
#col1 = [[220,30,16],[230,30,15],[80,12,12],[0,166,0],[16,15,14],[100,150,150]]
#col2 = [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]
#col3 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
#col4 = [[225,30,12],[230,26,15],[100,12,12],[0,166,10],[16,15,34],[107,150,150]]

#print(colorTest(col1,col1))
#print(colorTest(col2,col3))
#print(colorTest(col1,col4))