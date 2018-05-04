import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import set_link_color_palette

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1','ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.show()
 
row_dist = pd.DataFrame(squareform(
						pdist(df, metric='euclidean')),
						columns=labels, index=labels)

row_clusters = linkage(pdist(df, metric='euclidean'),
						method='complete')

cols = ['row label 1', 'row label 2', 'distance', 'no of items in cluster']
show = pd.DataFrame(row_clusters,
			columns=cols,
			index=['cluster %d' %(i+1) for i in 
					range(row_clusters.shape[0])])

print(show)

set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters,labels=labels,
						color_threshold=np.inf)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()

fig = plt.figure(figsize=(8,8), facecolor="white")
axd = fig.add_axes([0.09,0.1,0.2,0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23,0.1,0.6,0.6])
cax = axm.matshow(df_rowclust,
					interpolation='nearest', cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
	i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

ac = AgglomerativeClustering(n_clusters=3,
							affinity='euclidean',
							linkage='complete')
labels = ac.fit_predict(X)
print('Cluster Label: %s' %labels)

ac = AgglomerativeClustering(n_clusters=2,
							affinity='euclidean',
							linkage='complete')
labels = ac.fit_predict(X)
print('Cluster Label: %s' %labels)