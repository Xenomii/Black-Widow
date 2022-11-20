from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

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
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_scores = classifier.predict_proba(X_test)

    fig = plt.figure(figsize=(19, 10))
    tree.plot_tree(classifier, feature_names=['Response Code', 'User Agent', 'HTTP Request Type', 'URL Path'], class_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'], filled=True)
    plt.show()

    return y_pred, y_scores

def accuracy(y_test, y_pred):
    print("Model Accuracy: ", metrics.accuracy_score(y_test, y_pred))

def get_confusion(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    return cm

def get_accuracy(y_test, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)