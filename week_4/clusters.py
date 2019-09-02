import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage

def silhouette(X):
    scores = []
    for n in range(2,20):
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(X)
        labels = kmeans.labels_
        # preds = kmeans.fit_predict(X)
        scores.append(silhouette_score(X, labels))

    return round(max(scores),2)

def aglomerate(X):
    Z = linkage(X, method='average', metric='cosine')
    fcluster(Z, t=5, criterion='maxclust')
    dendrogram(Z)
    plt.show()
    # return dendrogram(Z)





def main():
    df = pd.read_csv('weather-check.csv')
    df = df.drop(labels=df.columns[0], axis=1)
    df = df.drop(labels=df.columns[2], axis=1)
    df_dummy = pd.get_dummies(df)
    print(df.head())
    X = df_dummy.values
    # print(silhouette(X))
    print(aglomerate(X))

if __name__ == '__main__':
    main()