import pickle
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve
from classifiers import run_classifiers, results_to_df, evaluation_metrics
import warnings

warnings.filterwarnings("ignore", category=Warning)


def run_RF(X_train, X_test, y_train, y_test, print_ROC=False):
    classifier = RandomForestClassifier(n_jobs=2, random_state=1)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    print("\n--- Time taken %s seconds ---\n" % (time.time() - start_time))

    classifier_name = "Random Forest"
    # save the model to disk
    filename = 'pretrained_models/%s_pretrained.sav' % classifier_name
    # pickle.dump(classifier, open(filename, 'wb'))

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


############################################# BOT EVOLUTION ################################################

def train_on_original_and_test_on_augmented_data():
    print('\n ~~~~~~~~~~~ Train on Original and Test on Augmented Data ~~~~~~~~~~~~~\n')

    df = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    synthetic_test_data = pickle.load(
        open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data', 'rb'))

    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_train = df['label']
    y_test = test_data['label']

    # Drop label column
    df = df.drop(['label'], axis=1)
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_train = df
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    df = results_to_df(acc, prec, f1, rec, g_m)
    df.to_csv('./multiclass_results/train_on_original_and_test_on_augmented_data_results.csv')


def train_and_test_on_augmented_data(train_ac_gan=False, test_ac_gan=False):
    print('\n ~~~~~~~~~~~ Train on Augmented and Testing on Augmented Data ~~~~~~~~~~~~~\n')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if train_ac_gan:
        synthetic_data_train = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_data_30K_per_class', 'rb'))
        train_filename = 'ac_gan'
    else:
        synthetic_data_train = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_30K_per_class', 'rb'))
        train_filename = 'cgan'

    if test_ac_gan:
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_test_data', 'rb'))
        test_filename = 'ac_gan'
    else:
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data', 'rb'))
        test_filename = 'cgan'

    augmented_df = original_train_data.append(synthetic_data_train)
    augmented_df = augmented_df.sample(frac=1)

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
    acc, prec, f1, rec, g_m = run_classifiers(X_train, X_test, y_train, y_test)

    df = results_to_df(acc, prec, f1, rec, g_m)
    df.to_csv('./multiclass_results/train_on_augmented_' + train_filename +
              '_and_test_on_augmented_' + test_filename + '_data.csv')

# Test on Augmented Data
# train_and_test_on_augmented_data(train_ac_gan=True, test_ac_gan=False)
# train_on_original_and_test_on_augmented_data()
