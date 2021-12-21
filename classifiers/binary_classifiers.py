import pickle
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.metrics import geometric_mean_score
import warnings

warnings.filterwarnings("ignore", category=Warning)


def run_classifiers(X_train, X_test, y_train, y_test):
    for classifier_name in ["bayes", "logreg", "rand_forest", "svm", "voting", "bagging", "adaboost"]:
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
        print("Accuracy {:.5f}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Precision {:.5f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
        print("F1-score {:.5f}".format(metrics.f1_score(y_test, y_pred, average='macro')))
        print("Recall-score {:.5f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
        print("G-Mean {:.5f}".format(geometric_mean_score(y_test, y_pred, correction=0.0001)))
        print("==============================\n")


def train_and_test_on_original_data():
    print('~~~~~~~~~~ Train and Test on Original Data')
    train_data = pickle.load(open('../data/original_data/train_binary_data', 'rb'))
    test_data = pickle.load(open('../data/original_data/test_binary_data', 'rb'))

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

    # ADASYN
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

    # summarize class distribution
    print("\nLabel distribution before ADASYN: {}".format(Counter(y_train)))
    print("Label distribution after ADASYN: {}\n".format(Counter(y_adasyn)))

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)

    # print('\n~~~~~~~~ Running Classifiers with ADASYN ~~~~~~~~~~~~~~~')
    # run_classifiers(X_adasyn, X_test, y_adasyn, y_test)


def train_on_original_and_test_on_augmented_data(cgan=False):
    train_original_data = pickle.load(open('../data/original_data/train_binary_data', 'rb'))
    original_test_data = pickle.load(open('../data/original_data/test_binary_data', 'rb'))

    if cgan:
        print('\n ~~~~~~~~~~~ Train on Original and Test on Augmented CGAN Data ~~~~~~~~~~~~~\n')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan/synthetic_binary_test_data', 'rb'))
    else:
        print('\n ~~~~~~~~~~~ Train on Original and Test on Augmented GAN Data ~~~~~~~~~~~~~\n')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/gan/synthetic_binary_test_data', 'rb'))

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_train = train_original_data['label']
    y_test = test_data['label']

    # Drop label column
    train_original_data = train_original_data.drop(['label'], axis=1)
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_train = train_original_data
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)


def train_on_augmented_and_test_on_original_data(cgan=False):
    # print('\n---------------- Training with 30000 new synthetic samples for each class  -------------------')
    train_original_data = pickle.load(open('../data/original_data/train_binary_data', 'rb'))

    if cgan:
        print('\n ~~~~~~~~~~~~~~~ Train with Augmented CGAN Data and Test on Original ~~~~~~~~~~~~~~~~')
        synthetic_data = pickle.load(
            open('../data/synthetic_data/conditional_gan/synthetic_binary_train_data', 'rb'))
    else:
        print('\n ~~~~~~~~~~~~~~~ Train with Augmented GAN Data and Test on Original ~~~~~~~~~~~~~~~~')
        synthetic_data = pickle.load(
            open('../data/synthetic_data/simple_gan/synthetic_binary_train_data', 'rb'))

    synthetic_data = synthetic_data.sample(frac=1)
    augmented_df = train_original_data.append(synthetic_data)
    augmented_df = augmented_df.sample(frac=1)

    y = augmented_df['label']
    print(augmented_df['label'].value_counts())

    # Drop unwanted columns
    augmented_df = augmented_df.drop(['label'], axis=1)

    X_train = augmented_df
    y_train = y

    test_df = pickle.load(open('../data/original_data/test_binary_data', 'rb'))

    y_test = test_df['label']
    print(test_df['label'].value_counts())

    # Drop unwanted columns
    test_df = test_df.drop(['label'], axis=1)

    X_test = test_df
    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # summarize class distribution
    print("\nLabel distribution: {}".format(Counter(y_train)))

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)


def train_and_test_on_augmented_data(train_cgan=False, test_cgan=False):

    if train_cgan:
        synthetic_data = pickle.load(
            open('../data/synthetic_data/conditional_gan/synthetic_binary_train_data', 'rb'))
        if test_cgan:
            print('\n ~~~~~~~~~~~ Train on Augmented CGAN and Test on Augmented CGAN Data ~~~~~~~~~~~~~\n')
            synthetic_test_data = pickle.load(
                open('../data/synthetic_data/conditional_gan/synthetic_binary_test_data', 'rb'))
        else:
            print('\n ~~~~~~~~~~~ Train on Augmented CGAN and Test on Augmented GAN Data ~~~~~~~~~~~~~\n')
            synthetic_test_data = pickle.load(
                open('../data/synthetic_data/gan/synthetic_binary_test_data', 'rb'))
    else:
        synthetic_data = pickle.load(
            open('../data/synthetic_data/gan/synthetic_binary_train_data', 'rb'))
        if test_cgan:
            print('\n ~~~~~~~~~~~ Train on Augmented GAN and Test on Augmented CGAN Data ~~~~~~~~~~~~~\n')
            synthetic_test_data = pickle.load(
                open('../data/synthetic_data/conditional_gan/synthetic_binary_test_data', 'rb'))
        else:
            print('\n ~~~~~~~~~~~ Train on Augmented GAN and Test on Augmented GAN Data ~~~~~~~~~~~~~\n')
            synthetic_test_data = pickle.load(
                open('../data/synthetic_data/gan/synthetic_binary_test_data', 'rb'))

    train_original_data = pickle.load(open('../data/original_data/train_binary_data', 'rb'))
    synthetic_data_train = synthetic_data.sample(frac=1)
    augmented_df = train_original_data.append(synthetic_data_train)
    augmented_df = augmented_df.sample(frac=1)

    original_test_data = pickle.load(open('../data/original_data/test_binary_data', 'rb'))
    print(original_test_data['label'].value_counts())

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_train = augmented_df['label']
    y_test = test_data['label']
    print(augmented_df['label'].value_counts())
    print(original_test_data['label'].value_counts())

    # Drop label column
    augmented_df = augmented_df.drop(['label'], axis=1)
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_train = augmented_df
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)


print('------- Binary Bot Detection --------')
# Train on Augmented Data
# train_and_test_on_augmented_data(cgan=True)
# train_on_augmented_and_test_on_original_data(cgan=True)

# Train on Original Data
# train_and_test_on_original_data()
train_on_original_and_test_on_augmented_data(cgan=False)
