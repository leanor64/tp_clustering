"""
Created on 2023/09/11

@author: huguet
"""

import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="diamond.arff"

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


# Run clustering method for a given number of clusters
# print("------------------------------------------------------")
# print("Appel KMeans pour une valeur de k fixée")
# tps1 = time.time()
# k=3
# model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
# model.fit(datanp)
# tps2 = time.time()
# labels = model.labels_
# # informations sur le clustering obtenu
# iteration = model.n_iter_
# inertie = model.inertia_
# centroids = model.cluster_centers_




#---------------------------------Pour déterminer le coefficient de silhouette en fonction de k
coeff_silhouette = []
indice_db = []
indice_ch = []
for k in range(2,20):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    #silhouette = silhouette_score(datanp, labels)
    #db = davies_bouldin_score(datanp, labels)
    ch = calinski_harabasz_score(datanp, labels)
   
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    centroids = model.cluster_centers_

    #coeff_silhouette.append(silhouette)
    #indice_db.append(db)
    indice_ch.append(ch)

#result = np.argmax(coeff_silhouette) + 2
#result = np.argmin(indice_db) + 2
result = np.argmax(indice_ch) + 2


#----------------------------------Graphique pour trouver le meilleur nombre de clusters - Silhouette
# plt.plot(range(2,20), coeff_silhouette, color="pink", marker="o")
# plt.xlabel("k")
# plt.ylabel("Coefficient de silhouette")
# plt.xticks(range(2,20))
# plt.suptitle(f"{name} - Coefficient de silhouette en fonction de k")
# plt.title(f"Meilleur k : {result}")
# plt.show()

#----------------------------------Graphique pour trouver le meilleur nombre de clusters - Davies Bouldin
# plt.plot(range(2,20), indice_db, color="pink", marker="o")
# plt.xlabel("k")
# plt.ylabel("Indice de Davies Bouldin")
# plt.xticks(range(2,20))
# plt.suptitle(f"{name} - Indice de Davies Bouldin en fonction de k")
# plt.title(f"Meilleur k : {result}")
# plt.show()

#----------------------------------Graphique pour trouver le meilleur nombre de clusters - Calinski Harabasz
plt.plot(range(2,20), indice_ch, color="pink", marker="o")
plt.xlabel("k")
plt.ylabel("Indice de Calinski Harabasz")
plt.xticks(range(2,20))
plt.suptitle(f"{name} - Indice de Calinski Harabasz en fonction de k")
plt.title(f"Meilleur k : {result}")
plt.show()



# ---------------------------------Pour déterminer l'inertie en fonction du nombre de clusters
# #inerties = []
# for k in range(1,20):
#     model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
#     model.fit(datanp)
#     tps2 = time.time()
#     labels = model.labels_
#     # informations sur le clustering obtenu
#     iteration = model.n_iter_
#     inertie = model.inertia_
#     centroids = model.cluster_centers_

#     inerties.append(inertie)

# -----------------------------------Graphique pour trouver le pt d'inflexion et déterminer le meilleur nombre de clusters
# plt.plot(range(1,20), inerties, color="pink", marker="o")
# plt.xlabel("k")
# plt.ylabel("inertie du clustering")
# plt.xticks(range(1,20))
# plt.title(f"{name} - Inertie du clustering en fonction de k")
# plt.show()

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)

