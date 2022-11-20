from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import math

def feature_scaling(X):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X

def split_train_test(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    # Get the square root of the test variables
    k = math.sqrt(len(y_test))
    if k % 2 == 0:
        k = k + 1

    # Training and running KNN
    neigh = KNeighborsClassifier(n_neighbors=int(k))
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    y_scores = neigh.predict_proba(X_test)
    return y_pred, y_scores

def get_confusion(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    return cm

def get_accuracy(y_test, y_pred):
    return metrics.accuracy_score(y_test, y_pred)
