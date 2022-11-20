def supervised_algo_results(accuracy=None, confusion=None, y_test=None, y_pred=None, TP=None, FP=None, FN=None, TN=None, Precision=None, Recall=None, Algo=None):
    message = "Algorithm Training and Testing Completed for " + Algo + "!\n\n"
    message += "Model Details\n"
    message += "=========================================\n"
    message += "Model Accuracy: " + str(accuracy) + "\n\n"
    message += "Confusion Matrix:\n" + str(confusion) + "\n\n"
    message += "True Positive:\n" + str(TP) + "\n\n"
    message += "False Positive:\n" + str(FP) + "\n\n"
    message += "False Negative:\n" + str(FN) + "\n\n"
    message += "True Negative:\n" + str(TN) + "\n\n"
    message += "Precision:\n" + str(Precision) + "\n\n"
    message += "Recall:\n" + str(Recall) + "\n\n"
    message += "=========================================\n"
    message += "Testing Data Vs Predictions\n"
    message += "=========================================\n\n"
    message += "Actual Values: " + str(y_test) + "\n\n"
    message += "Predicted Values: " + str(y_pred)
    return message

def kmeans_results(pred_cluster, accuracy):
    message = "K-Means Clustering Completed!\n\n"
    message += "Model Details\n"
    message += "=========================================\n"
    message += "Model Accuracy: " + str(accuracy) + "\n\n"
    message += "Cluster Prediction: " + str(pred_cluster) + "\n"
    message += "=========================================\n"
    return message

def dbscan_results(n_clusters, n_noise):
    message = "DBSCAN Clustering Completed!\n\n"
    message += "Model Details\n"
    message += "=========================================\n"
    message += "Number of Clusters: " + str(n_clusters) + "\n\n"
    message += "Number of Noise Points: " + str(n_noise) + "\n"
    message += "=========================================\n"
    return message
    