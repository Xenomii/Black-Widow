import sys
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

def feature_scaling(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X

def get_wcss(x):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    return wcss

def apply_kmeans(x, dataframe):
    # Applying kmeans to the dataset / Creating the kmeans classifier
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)
    accuracy = metrics.adjusted_rand_score(y_kmeans, dataframe["severity"])
    np.set_printoptions(threshold=sys.maxsize)
    return y_kmeans, kmeans, accuracy