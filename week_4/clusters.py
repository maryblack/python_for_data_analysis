import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn.cluster import DBSCAN

def silhouette(df):
    df_dummy = pd.get_dummies(df)
    # print(df.info())
    X = df_dummy.values
    scores = []
    for n in range(2,20):
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(X)
        labels = kmeans.labels_
        # preds = kmeans.fit_predict(X)
        scores.append(silhouette_score(X, labels))

    return max(scores)

def aglomerate(df):
    df_dummy = pd.get_dummies(df)
    X = df_dummy.values
    # X = df_dummy.iloc[:, 1:].values
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    Z = linkage(X, method='average', metric='cosine')
    label = fcluster(Z, t=5, criterion='maxclust')
    df.loc[:, 'label'] = label
    for i, group in df.groupby('label'):
        print('=' *10)
        print('cluster{}'.format(i))
        print(len(group))
        print(group['What is your gender?'].value_counts(normalize=True))

    plt.show()
    # return dendrogram(Z)

def dbscen_ex(df):
    # df = df.drop(labels=df.columns[0], axis=1)
    df_dummy = pd.get_dummies(df)
    # print(df.info())
    X = df_dummy.values
    df.replace('-', np.nan, inplace=True)
    print(len(df.index))
    df1 = df.dropna()
    print(f'withoun NaN = {len(df1.index)}')
    e_m = np.linspace(0.1, 1.0, num = 10)
    for e in e_m:
        db = DBSCAN(eps=e, min_samples=20, metric='cosine').fit(X)
        label = db.labels_
        df.loc[:, 'label'] = label
        # print(label)
        for i, group in df.groupby('label'):
            print('=' * 10)
            print('cluster{}'.format(i))
            print('eps={}'.format(e))
            print(group)


def main():
    df = pd.read_csv('weather-check.csv')
    df = df.drop(labels=df.columns[0], axis=1)
    df = df.drop(labels=df.columns[2], axis=1)
    print(silhouette(df))
    print(aglomerate(df))
    print(dbscen_ex(df))

if __name__ == '__main__':
    main()