from modules import knn as knn
from modules import kmeans as kmeans
from modules import dbscan as dbscan
from modules import decisiontree as decision
import numpy as np
import pandas as pd


def get_confusion_matrix_values_results(cm):
   FP = pd.Series(cm.sum(axis=0) - np.diag(cm), index=["Severity 0", "Severity 1", "Severity 2", "Severity 3"])
   FN = pd.Series(cm.sum(axis=1) - np.diag(cm), index=["Severity 0", "Severity 1", "Severity 2", "Severity 3"])
   TP = pd.Series(np.diag(cm), index=["Severity 0", "Severity 1", "Severity 2", "Severity 3"])
   TN = pd.Series(np.matrix(cm).sum() - (FP + FN + TP), index=["Severity 0", "Severity 1", "Severity 2", "Severity 3"])

   Precision = TP/(TP+FP)
   Recall = TP/(TP+FN)
   Accuracy = (TP+TN)/(TP+TN+FP+FN)

   return TP, FP, FN, TN, Precision, Recall, Accuracy

def run_knn(dataframe, progressBar):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    progressBar.setValue(25)
    X = knn.feature_scaling(X)
    X_train, X_test, y_train, y_test = knn.split_train_test(X, y)
    y_pred, y_scores = knn.train_model(X_train, y_train, X_test, y_test)
    confusion = knn.get_confusion(y_test, y_pred)
    accuracy = knn.get_accuracy(y_test, y_pred)
    TP, FP, FN, TN, Precision, Recall, Accuracy = get_confusion_matrix_values_results(confusion)

    return accuracy, confusion, X_test, y_test, y_pred, TP, FP, FN, TN, Precision, Recall, Accuracy, confusion, y_scores, X ,y

def run_kmeans(dataframe, progressBar):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    progressBar.setValue(25)

    X = kmeans.feature_scaling(X)
    wcss = kmeans.get_wcss(X)
    y_kmean, kmeans_classifier, accuracy = kmeans.apply_kmeans(X, dataframe)

    return y_kmean, X, kmeans_classifier, wcss, accuracy

def run_dbscan(dataframe, progressBar):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    progressBar.setValue(25)
    X = dbscan.feature_scaling(X)
    core_samples_mask, labels = dbscan.compute_dbscan(X, 0.5, 10)
    n_clusters, n_noise = dbscan.get_details(labels)

    return X, labels, n_clusters, n_noise, core_samples_mask

def run_decision(dataframe, progressBar):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    progressBar.setValue(25)
    # X = decision.feature_scaling(X)
    X_train, X_test, y_train, y_test = decision.split_train_test(X, y)
    y_pred, y_scores = decision.train_model(X_train, y_train, X_test, y_test)
    confusion = decision.get_confusion(y_test, y_pred)
    accuracy = decision.get_accuracy(y_test, y_pred)
    TP, FP, FN, TN, Precision, Recall, Accuracy = get_confusion_matrix_values_results(confusion)

    return accuracy, confusion, X_test, y_test, y_pred, TP, FP, FN, TN, Precision, Recall, confusion, y_scores