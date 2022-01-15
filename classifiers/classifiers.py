import pickle
import time
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.metrics import geometric_mean_score
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore", category=Warning)


def run_classifiers(X_train, X_test, y_train, y_test):
    acc = []
    prec = []
    f1 = []
    rec = []
    g_m = []
    for classifier_name in ["Naive Bayes", "SVM", "MLP", "AdaBoost", "Random Forest"]:
        if classifier_name == "MLP":
            classifier = MLPClassifier(random_state=1)
        elif classifier_name == "Naive Bayes":
            classifier = MultinomialNB()
        elif classifier_name == "Random Forest":
            classifier = RandomForestClassifier(n_jobs=-1, random_state=1)
        elif classifier_name == "SVM":
            classifier = LinearSVC()
        elif classifier_name == "AdaBoost":
            classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_jobs=-1, random_state=1),
                                            random_state=1)
        else:
            raise ValueError(f"Unknown classifier {classifier_name}")

        start_time = time.time()
        classifier.fit(X_train, y_train)
        print("\n--- Time taken %s seconds ---\n" % (time.time() - start_time))

        # save the model to disk
        filename = 'pretrained_models/%s_pretrained.sav' % classifier_name
        # pickle.dump(classifier, open(filename, 'wb'))

        y_pred = classifier.predict(X_test)

        print("==============================")

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 5)
        acc.append(accuracy)
        precision = round(metrics.precision_score(y_test, y_pred, average='macro'), 5)
        prec.append(precision)
        f1_score = round(metrics.f1_score(y_test, y_pred, average='macro'), 5)
        f1.append(f1_score)
        recall = round(metrics.recall_score(y_test, y_pred, average='macro'), 5)
        rec.append(recall)
        g_mean = round(geometric_mean_score(y_test, y_pred, correction=0.0001), 5)
        g_m.append(g_mean)

        print('Results for {:}'.format(classifier))
        print("Accuracy {:.5f}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Precision {:.5f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
        print("F1-score {:.5f}".format(metrics.f1_score(y_test, y_pred, average='macro')))
        print("Recall-score {:.5f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
        print("G-Mean {:.5f}".format(geometric_mean_score(y_test, y_pred, correction=0.0001)))
        print("==============================\n")

    return acc, prec, f1, rec, g_m


def results_to_df(accuracy, precision, f1_score, recall, g_mean, auc_score, pr_score, rf=False):
    if rf:
        df = pd.DataFrame(columns=["rand_forest"])
        acc_series = pd.Series(accuracy, index=df.columns)
        prec_series = pd.Series(precision, index=df.columns)
        f1_series = pd.Series(f1_score, index=df.columns)
        rec_series = pd.Series(recall, index=df.columns)
        gm_series = pd.Series(g_mean, index=df.columns)
        auc_series = pd.Series(auc_score, index=df.columns)
        pr_series = pd.Series(pr_score, index=df.columns)

        df = df.append(acc_series, ignore_index=True)
        df = df.append(prec_series, ignore_index=True)
        df = df.append(f1_series, ignore_index=True)
        df = df.append(rec_series, ignore_index=True)
        df = df.append(gm_series, ignore_index=True)
        df = df.append(auc_series, ignore_index=True)
        df = df.append(pr_series, ignore_index=True)

        df.index = ['Accuracy', 'Precision', 'F1 score', 'Recall', 'G-Mean', 'AUC Score', 'PR Score']

    else:
        df = pd.DataFrame(columns=["bayes", "svm", "mlp", "adaboost", "rand_forest"])
        acc_series = pd.Series(accuracy, index=df.columns)
        prec_series = pd.Series(precision, index=df.columns)
        f1_series = pd.Series(f1_score, index=df.columns)
        rec_series = pd.Series(recall, index=df.columns)
        gm_series = pd.Series(g_mean, index=df.columns)

        df = df.append(acc_series, ignore_index=True)
        df = df.append(prec_series, ignore_index=True)
        df = df.append(f1_series, ignore_index=True)
        df = df.append(rec_series, ignore_index=True)
        df = df.append(gm_series, ignore_index=True)

        df.index = ['Accuracy', 'Precision', 'F1 score', 'Recall', 'G-Mean']

    return df


def evaluation_metrics(classifier, y_test, y_pred):
    acc = []
    prec = []
    f1 = []
    rec = []
    g_m = []
    accuracy = round(metrics.accuracy_score(y_test, y_pred), 5)
    acc.append(accuracy)
    precision = round(metrics.precision_score(y_test, y_pred, average='macro'), 5)
    prec.append(precision)
    f1_score = round(metrics.f1_score(y_test, y_pred, average='macro'), 5)
    f1.append(f1_score)
    recall = round(metrics.recall_score(y_test, y_pred, average='macro'), 5)
    rec.append(recall)
    g_mean = round(geometric_mean_score(y_test, y_pred, correction=0.0001), 5)
    g_m.append(g_mean)

    print('Results for {:}'.format(classifier))
    print("Accuracy {:.5f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Precision {:.5f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
    print("F1-score {:.5f}".format(metrics.f1_score(y_test, y_pred, average='macro')))
    print("Recall-score {:.5f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
    print("G-Mean {:.5f}".format(geometric_mean_score(y_test, y_pred, correction=0.0001)))
    print("==============================\n")
    return acc, prec, f1, rec, g_m
