import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

#opts
pd.set_option('display.max_columns', None)
def print_full(x):
    pd.set_option('display.max_colwidth', None) 
    print(x)
    pd.reset_option('display.max_colwidth')



df = pd.read_csv("sampleExtraction.csv")

newDf = df[["family-income","age"]]
residency_type = []
for i in range(len(df)):
    if (df.iloc[i]["residency-type"] == "Rural"):
        residency_type.append(1)
    elif df.iloc[i]["residency-type"] == "SubUrban":
        residency_type.append(2)
    else:
        residency_type.append(3)

print(residency_type)
df["residency-typeNum"] = residency_type
print(df.head())
x = df["family-income"]
y = df["age"]
z = df["residency-typeNum"]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
ax.set_xlabel("Family Income")
ax.set_ylabel("Age")
ax.set_zlabel("Residency Type")
plt.title("Visualization of Data")
plt.show() 


##We have the plot but now we want to know what the optional amount of clusters are
#use the elbow method initially

#Data needs to be triplets
x_data = []
for i, (x, y, z) in enumerate(zip(x, y, z)):
    x_data.append([x,y,z])

print(x_data)

wcss = []
k_val = []
for k in range(1,11):
    print(f"{k} in progress")
    k_means = KMeans(n_clusters = k, init = 'k-means++')
    k_means.fit(x_data)
    wcss.append(k_means.inertia_)
    k_val.append(k)

#do a plot to identify the elbow (dont want to overfit)

plt.plot(k_val,wcss)
plt.xlabel("k")
plt.ylabel("wcss")
plt.title("wcss vs k")
plt.show()

#elbow at k=2, visually though it seems that we might have more than two. Do a better silhouette score next

#silhouette score
silhouette = []
k_val = []
for k in range(2,11):
    print(f"{k} in progress")
    k_means = KMeans(n_clusters = k, init = 'k-means++')
    k_means.fit(x_data)
    silhouette.append(silhouette_score(x_data, k_means.labels_))
    k_val.append(k)

plt.plot(k_val,silhouette)
plt.xlabel("k")
plt.ylabel("silhouette")
plt.title("silhouette score vs k")
plt.show()

#represents k=5 as the best

k_means_optimum = KMeans(n_clusters = 5, init = 'k-means++')
y = k_means_optimum.fit_predict(x_data)
print(y)


#separate data
df["k"] = y
cluster1 = df[df.k==0]
cluster2 = df[df.k==1]
cluster3 = df[df.k==2]
cluster4 = df[df.k==3]
cluster5 = df[df.k==4]

#display clusters

kplot = plt.axes(projection='3d')
# Data for three-dimensional scattered points
kplot.scatter3D(cluster1["family-income"], cluster1["age"], cluster1["residency-typeNum"], c='red', label = 'Cluster 1')
kplot.scatter3D(cluster2["family-income"], cluster2["age"], cluster2["residency-typeNum"], c='blue', label = 'Cluster 2')
kplot.scatter3D(cluster3["family-income"], cluster3["age"], cluster3["residency-typeNum"], c='black', label = 'Cluster 3')
kplot.scatter3D(cluster4["family-income"], cluster4["age"], cluster4["residency-typeNum"], c='purple', label = 'Cluster 4')
kplot.scatter3D(cluster5["family-income"], cluster5["age"], cluster5["residency-typeNum"], c='orange', label = 'Cluster 5')
kplot.scatter3D(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], k_means_optimum.cluster_centers_[:,2] , color = 'indigo', s = 200)
plt.legend()
plt.xlabel("Family Income")
plt.ylabel("Age")
kplot.set_zlabel("Residency Type")
plt.title("Visualization of Clusters")
plt.show()

#samples from each cluster
clusters = [cluster1,cluster2,cluster3,cluster4,cluster5]

i=1
for cluster in clusters:
    print(f"CLUSTER {i}")
    print_full(cluster["reason"])
    print(cluster[["family-income","residency-type","age"]])
    i+=1
   
