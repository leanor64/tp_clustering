import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="banana.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Distances aux k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors(n_neighbors = k)
neigh.fit(datanp)
distances , indices = neigh.kneighbors(datanp)
# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np.asarray ( [ np.average(distances [ i ] [ 1 : ] ) for i in range (0,distances.shape [ 0 ] ) ])
# trier par ordre croissant
distancetrie = np.sort ( newDistances )
plt.title ( " Plus proches voisins " + str (k) )
plt.plot( distancetrie )
plt.show ()



# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
min_pts=5 
table_sil = []

#-------------------Pour déterminer la meilleure valeur de epsilon----------

trange = [0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.020,0.022,0.024]
for epsilon in trange:
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    sil = metrics.silhouette_score(datanp, labels)
    table_sil.append(sil)

result = trange[np.argmax(table_sil)]
plt.plot(trange, table_sil, color="red", marker="o")
plt.xlabel("epsilon")
plt.ylabel("Coefficient de silhouette")
plt.xticks(trange)
plt.suptitle(f"{name} - Coefficient de silhouette en fonction de epsilon")
plt.title(f"Meilleur epsilon : {result}")
plt.show()

#------Pour epsilon = 0.014

epsilon = 0.014
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

#Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('----------------------------------------------------')
print(f'for epsilon = {epsilon} ')
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epsilon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()

# --------Pour epsilon = 0.016

epsilon = 0.016
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

#Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('----------------------------------------------------')
print(f'for epsilon = {epsilon} ')
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epsilon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()


#------Pour epsilon = 0.020

epsilon = 0.020
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

#Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('----------------------------------------------------')
print(f'for epsilon = {epsilon} ')
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epsilon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()

