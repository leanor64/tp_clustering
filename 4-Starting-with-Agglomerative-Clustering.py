import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="xclara.arff"

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



### Pour fixer la distance
tps1 = time.time()
seuil_dist=40
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

#---------------------------------Pour déterminer le coefficient de silhouette en fonction de k
tps1 = time.time()
coeff_silhouette = []
table_k = []
for seuil_dist in range(20,70,5):
    model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    silhouette = metrics.silhouette_score(datanp, labels)
    coeff_silhouette.append(silhouette)
    # Nb iteration of this method
    #iteration = model.n_iter_
    k = model.n_clusters_
    table_k.append(k)
    leaves=model.n_leaves_

result = table_k[np.argmax(coeff_silhouette)]


plt.plot(table_k, coeff_silhouette, color="red", marker="o")
plt.xlabel("Nombre de cluster k")
plt.ylabel("Coefficient de silhouette")
plt.xticks(table_k)
plt.suptitle(f"{name} - Coefficient de silhouette en fonction de k")
plt.title(f"Meilleur k : {result}")
plt.show()

#---------------------------------Pour déterminer l'indice de CH en fonction de k
tps1 = time.time()
table_ch = []
table_k = []
for seuil_dist in range(20,70,5):
    model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    ch = metrics.calinski_harabasz_score(datanp, labels)
    table_ch.append(ch)
    # Nb iteration of this method
    #iteration = model.n_iter_
    k = model.n_clusters_
    table_k.append(k)
    leaves=model.n_leaves_

result = table_k[np.argmax(table_ch)]


plt.plot(table_k, table_ch, color="red", marker="o")
plt.xlabel("Nombre de cluster k")
plt.ylabel("Indice de Calinski Harabasz")
plt.xticks(table_k)
plt.suptitle(f"{name} - Indice de Calinski Harabasz en fonction de k")
plt.title(f"Meilleur k : {result}")
plt.show()


###
# FIXER le nombre de clusters
###
k=3
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

#---------------------------------Pour déterminer le coefficient de silhouette en fonction de k
table_sil = []
for k in range(2,20):
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    kres = model.n_clusters_
    leaves=model.n_leaves_
    sil = metrics.silhouette_score(datanp, labels)
    table_sil.append(sil)

result = np.argmax(table_sil) + 2
plt.plot(range(2,20), table_sil, color="red", marker="o")
plt.xlabel("Nombre de cluster k")
plt.ylabel("Coefficient de silhouette")
plt.xticks(range(2,20))
plt.suptitle(f"{name} - Coefficient de silhouette en fonction de k")
plt.title(f"Meilleur k : {result}")
plt.show()


#---------------------------------Pour déterminer l'indice de CH en fonction de k
table_ch = []
for k in range(2,20):
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    kres = model.n_clusters_
    leaves=model.n_leaves_
    ch = metrics.calinski_harabasz_score(datanp, labels)
    table_ch.append(ch)
result = np.argmax(table_ch) + 2

plt.plot(range(2,20), table_ch, color="red", marker="o")
plt.xlabel("Nombre de cluster k")
plt.ylabel("Indice de Calinski Harabasz")
plt.xticks(range(2,20))
plt.suptitle(f"{name} - Indice de Calinski Harabasz en fonction de k")
plt.title(f"Meilleur k : {result}")
plt.show()

#######################################################################