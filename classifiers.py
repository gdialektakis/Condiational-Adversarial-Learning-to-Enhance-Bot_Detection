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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import warnings


def run_classifiers(X_train, X_test, y_train, y_test):
    for classifier_name in ["voting", "bagging", "adaboost"]: #"bayes", "logreg", "rand_forest", "svm",
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

        y_pred_train = classifier.predict(X_train)
        y_pred = classifier.predict(X_test)

        print("==============================")
        print('Results for {:}'.format(classifier))
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred, average='macro'))
        print("F1-score: ", metrics.f1_score(y_test, y_pred, average='macro'))
        print("Recall-score: ", metrics.recall_score(y_test, y_pred, average='macro'))
        print("==============================\n")


def train_with_original_data():
    df = pickle.load(open('gans/multi_class_data', 'rb'))

    # Convert features that are boolean to integers
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    y = df['label']
    print(df['label'].value_counts())

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id', 'label'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    # split the labeled tweets into train and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df, y, test_size=0.2)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # SMOTE
    smote_sampler = SMOTE(sampling_strategy='not majority', n_jobs=-1)
    X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)

    # summarize class distribution
    print("\nLabel distribution before SMOTE: {}".format(Counter(y_train)))
    print("Label distribution after SMOTE: {}\n".format(Counter(y_smote)))

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)

    print('\n~~~~~~~~ Running Classifiers with SMOTE ~~~~~~~~~~~~~~~')
    run_classifiers(X_smote, X_test, y_smote, y_test)


# TODO
def train_with_augmented_data():
    df = pickle.load(open('gans/conditional_gan_multi/train_multiclass_data', 'rb'))
    synthetic_data = pickle.load(open('gans/conditional_gan_multi/synthetic_multiclass_data_30000_each_class', 'rb'))
    synthetic_data = synthetic_data.sample(frac=1)
    augmented_df = df.append(synthetic_data)
    augmented_df = augmented_df.sample(frac=1)
    # Convert features that are boolean to integers
    augmented_df = augmented_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    y = augmented_df['label']
    print(augmented_df['label'].value_counts())

    # Drop unwanted columns
    augmented_df = augmented_df.drop(['user_name', 'user_screen_name', 'user_id', 'label'], axis=1)
    if 'max_appearance_of_punc_mark' in augmented_df.columns:
        augmented_df = augmented_df.drop(['max_appearance_of_punc_mark'], axis=1)

    X_train = augmented_df
    y_train = y

    test_df = pickle.load(open('gans/conditional_gan_multi/test_multiclass_data', 'rb'))

    # Convert features that are boolean to integers
    test_df = test_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    y_test = test_df['label']
    print(test_df['label'].value_counts())

    # Drop unwanted columns
    test_df = test_df.drop(['user_name', 'user_screen_name', 'user_id', 'label'], axis=1)
    if 'max_appearance_of_punc_mark' in test_df.columns:
        test_df = test_df.drop(['max_appearance_of_punc_mark'], axis=1)

    X_test = test_df
    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # SMOTE
    smote_sampler = SMOTE(sampling_strategy='not majority', n_jobs=-1)
    X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)

    # summarize class distribution
    print("\nLabel distribution before SMOTE: {}".format(Counter(y_train)))
    print("Label distribution after SMOTE: {}\n".format(Counter(y_smote)))

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)

    #print('\n~~~~~~~~ Running Classifiers with SMOTE ~~~~~~~~~~~~~~~')
    #run_classifiers(X_smote, X_test, y_smote, y_test)


# TODO
def train_with_original_and_test_on_synthetic_data():
    df = pickle.load(open('gans/multi_class_data', 'rb'))
    synthetic_data = pickle.load(open('gans/conditional_gan_multi/synthetic_multiclass_data_30000_each_class', 'rb'))
    synthetic_data = synthetic_data.sample(frac=1)
    # Convert features that are boolean to integers
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    y = df['label']
    print(df['label'].value_counts())

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id', 'label'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    # split the labeled tweets into train and test set
    X_train = df
    y_train = y

    # Convert features that are boolean to integers
    synthetic_data = synthetic_data.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    y_test = synthetic_data['label']

    # Drop unwanted columns
    synthetic_data = synthetic_data.drop(['label'], axis=1)
    X_test = synthetic_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # SMOTE
    smote_sampler = SMOTE(sampling_strategy='not majority', n_jobs=-1)
    X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)

    # summarize class distribution
    print("\nLabel distribution before SMOTE: {}".format(Counter(y_train)))
    print("Label distribution after SMOTE: {}\n".format(Counter(y_smote)))

    print('\n~~~~~~~~ Running Classifiers ~~~~~~~~~~~~~~~')
    run_classifiers(X_train, X_test, y_train, y_test)

    print('\n~~~~~~~~ Running Classifiers with SMOTE ~~~~~~~~~~~~~~~')
    run_classifiers(X_smote, X_test, y_smote, y_test)


#train_with_original_data()
#train_with_original_and_test_on_synthetic_data()
train_with_augmented_data()
