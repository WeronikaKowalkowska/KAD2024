import learn as sk
import numpy as np
import pandas as pd
#from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
#import seaborn as sns
#%matplotlib inline
#import os
#os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


dane = pd.read_csv('data2.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
kmeans=cluster.KMeans(n_clusters=3,init='k-means++')
#kmeans=kmeans.fit(dane)
kmeans=kmeans.fit(dane[['Sepal length','Petal length']])
kmeans.cluster_centers_
dane['Clusters']=kmeans.labels_
dane['Clusters'].value_counts()
#dane.to_csv("clusters.csv",index=False)
#sns.scatterplot(x="Sepal lenght", y="Petal lenght", hue="Clusters", data=dane,)
plt.scatter(x=dane['Sepal length'], y=dane['Petal length'], c=dane['Clusters'], marker='o')
plt.title("KMeans Clustering")
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.colorbar(label='Cluster')
plt.show()