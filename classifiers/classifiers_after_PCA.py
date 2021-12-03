import pickle
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.metrics import geometric_mean_score
from helper_scripts.dimensionality_reduction import dimensionality_reduction
import warnings

warnings.filterwarnings("ignore", category=Warning)


def run_classifiers(X_train, X_test, y_train, y_test):
    for classifier_name in ["logreg", "rand_forest", "svm", "voting", "bagging", "adaboost"]:
        if classifier_name == "logreg":
            classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif classifier_name == "bayes":
            classifier = MultinomialNB(alpha=0.01)
        elif classifier_name == "rand_forest":
            classifier = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
        elif classifier_name == "svm":
            classifier = LinearSVC()
        elif classifier_name == "bagging":
            classifier = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10)
        elif classifier_name == "adaboost":
            classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=50)
        elif classifier_name == "voting":
            clf1 = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf2 = MultinomialNB(alpha=0.01)
            clf3 = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
            classifier = VotingClassifier(estimators=[('lr', clf1), ('nb', clf2), ('rf', clf3)], voting='soft')
        else:
            raise ValueError(f"Unknown classifier {classifier_name}")

        start_time = time.time()
        classifier.fit(X_train, y_train)
        print("\n--- Time taken %s seconds ---\n" % (time.time() - start_time))

        # Plotting the learning curve for each classifier
        # Create the learning curve visualizer
        """
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        visualizer = LearningCurve(classifier, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)
        visualizer.fit(X_train, y_train)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure
        """
        # save the model to disk
        filename = 'pretrained_models/%s_pretrained.sav' % classifier_name
        # pickle.dump(classifier, open(filename, 'wb'))

        y_pred = classifier.predict(X_test)

        print("==============================")
        print('Results for {:}'.format(classifier))
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred, average='macro'))
        print("F1-score: ", metrics.f1_score(y_test, y_pred, average='macro'))
        print("Recall-score: ", metrics.recall_score(y_test, y_pred, average='macro'))
        print("G-Mean: ", geometric_mean_score(y_test, y_pred, correction=0.0001))
        print("==============================\n")


"""
    The following methods train classifiers on different version of data with only 6 features 
    (1 dimension for each feature category).
"""


def train_and_test_on_original_data():
    data = dimensionality_reduction()

    data = data.sample(frac=1)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    y_train = train_data['label']
    y_test = test_data['label']

    # Drop unwanted columns
    X_train = train_data.drop(['label'], axis=1)
    X_test = test_data.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)


train_and_test_on_original_data()
