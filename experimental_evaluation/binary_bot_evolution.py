import pickle
import time
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
import warnings

from yellowbrick import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve

from experimental_evaluation.classifiers import evaluation_metrics, run_classifiers, results_to_df

warnings.filterwarnings("ignore", category=Warning)


def run_RF(X_train, X_test, y_train, y_test, print_ROC=False):
    classifier = RandomForestClassifier(n_jobs=2, random_state=1)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    print("\n--- Time taken %s seconds ---\n" % (time.time() - start_time))

    classifier_name = "Random Forest"

    y_pred = classifier.predict(X_test)

    acc, prec, f1, rec, g_m = evaluation_metrics(classifier_name, y_test, y_pred)

    auc_score = 0
    pr_score = 0
    if print_ROC:
        visualizer = ROCAUC(classifier, is_fitted=True, per_class=False, micro=False, macro=True)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()  # Finalize and render the figure
        auc_score = visualizer.score(X_test, y_test)
        print("AUC score {:.5f} ".format(auc_score))

        viz = PrecisionRecallCurve(classifier, micro=True, per_class=True)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        pr_score = viz.score(X_test, y_test)
        print("Precision-Recall score {:.5f} ".format(pr_score))
        viz.show()

    return acc, prec, f1, rec, g_m, auc_score, pr_score


def train_on_old_data():

    print('~~~~~~~~~~ Train on Old Data ~~~~~~~~~~ ')
    train_data = pickle.load(open('../binary_data/train_old_data', 'rb'))
    test_data = pickle.load(open('../binary_data/test_old_data', 'rb'))

    print('Old data distribution')
    print(train_data['label'].value_counts())

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

    print('\n~~~~~~~~ Training Random Forest on old data and test on old data ~~~~~~~~~~~~~~~')
    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('train_and_test_on_old_data_results.csv')

    ########## Test on new data #########
    test_data_new = pickle.load(open('../binary_data/new_data', 'rb'))

    print('New data distribution')
    print(test_data_new['label'].value_counts())

    y_test_new = test_data_new['label']
    y_test_new = y_test_new.astype('int')

    X_test_new = test_data_new.drop(['label'], axis=1)
    X_test_new = scaler.transform(X=X_test_new)

    print('\n~~~~~~~~ Training Random Forest on old data and test on NEW data ~~~~~~~~~~~~~~~')
    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test_new, y_train, y_test_new, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('train_on_old_and_test_on_new_data_results.csv')


def train_on_augmented_data(cgan=True, ac_gan=False):
    print('~~~~~~~~~~ Train on Augmented Data')
    train_data = pickle.load(open('../binary_data/train_old_data', 'rb'))
    test_data = pickle.load(open('../binary_data/test_old_data', 'rb'))

    if cgan:
        print('\n ~~~~~~~~~~~~~~~ Train with Augmented CGAN Data ~~~~~~~~~~~~~~~~')
        synthetic_data = pickle.load(open('../binary_data/synthetic_data/cgan/synthetic_binary_data', 'rb'))
        filename = 'cgan'
    else:
        print('\n ~~~~~~~~~~~~~~~ Train with Augmented AC-GAN Data ~~~~~~~~~~~~~~~~')
        synthetic_data = pickle.load(open('../binary_data/synthetic_data/ac_gan/synthetic_binary_data', 'rb'))
        filename = 'ac_gan'

    synthetic_data = synthetic_data.sample(frac=1)
    augmented_data = train_data.append(synthetic_data)
    augmented_data = augmented_data.sample(frac=1)

    y_train = augmented_data['label']
    print(augmented_data['label'].value_counts())

    # Drop unwanted columns
    X_train = augmented_data.drop(['label'], axis=1)

    y_test = test_data['label']
    # Drop unwanted columns
    X_test = test_data.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    print('\n~~~~~~~~ Training Random Forest on Augmented data and test on old data ~~~~~~~~~~~~~~~')
    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('train_on_augmented_'+filename+'_and_test_on_old_data_results.csv')

    ########## Test on new data #########
    test_data_new = pickle.load(open('../binary_data/new_data', 'rb'))

    y_test_new = test_data_new['label']
    y_test_new = y_test_new.astype('int')

    X_test_new = test_data_new.drop(['label'], axis=1)
    X_test_new = scaler.transform(X=X_test_new)

    print('\n~~~~~~~~ Training Random Forest on Augmented data and test on NEW data ~~~~~~~~~~~~~~~')
    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test_new, y_train, y_test_new, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('train_on_augmented_'+filename+'_and_test_on_new_data_results.csv')


print('------- Binary Bot Detection --------')
train_on_old_data()
# Train on Augmented Data
# train_on_augmented_and_test_on_original_data(cgan=True)

# Train on Original Data
# train_and_test_on_original_data()
# train_on_original_and_test_on_augmented_data(cgan=True)
