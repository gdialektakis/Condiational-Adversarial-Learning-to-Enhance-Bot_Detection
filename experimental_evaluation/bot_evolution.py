import pickle
import time

from imblearn.over_sampling import ADASYN
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

def train_on_original_and_test_on_augmented_data(cgan, ac_gan, adasyn):
    print('\n ~~~~~~~~~~~ Train on Original and Test on Augmented Data ~~~~~~~~~~~~~\n')

    df = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))

    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if cgan:
        filename = 'cgan'
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
    elif ac_gan:
        filename = 'ac_gan'
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
    else:
        filename = 'adasyn'
        """
        adasyn = ADASYN(random_state=42, n_jobs=-1)
        X_test = original_test_data.drop(['label'], axis=1)
        y_test = original_test_data['label']
        X_adasyn, y_adasyn = adasyn.fit_resample(X_test, y_test)

        augmented_data = pd.DataFrame(data=X_adasyn, columns=X_test.columns)
        augmented_data['label'] = y_adasyn
        original_data = original_test_data
        synthetic_samples_adasyn = augmented_data[~augmented_data.isin(original_data)].dropna()
        pickle.dump(synthetic_samples_adasyn, open('synthetic_test_data_adasyn', 'wb'))
        """
        synthetic_test_data = pickle.load(open('synthetic_test_data_adasyn', 'rb'))

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    print('Total Test data label distribution')
    print(test_data['label'].value_counts())

    print('Synthetic Test data label distribution')
    print(synthetic_test_data['label'].value_counts())

    y_train = df['label']
    y_test = test_data['label']

    # Drop label column
    X_train = df.drop(['label'], axis=1)
    X_test = test_data.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=True)
    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('./bot_evolution/multi_class/train_on_original_and_test_on_augmented_' + filename + '_data_results.csv')


def train_on_adasyn_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=False):
    print('\n ~~~~~~~~~~~ Train on Augmented and Test on Augmented Data ~~~~~~~~~~~~~\n')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    y_train = original_train_data['label']
    X_train = original_train_data.drop(['label'], axis=1)

    adasyn = ADASYN(random_state=42, n_jobs=-1)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
    print(y_adasyn.value_counts(normalize=False))

    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if test_cgan:
        print('Testing on CGAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
        filename = 'cgan'
    elif test_ac_gan:
        print('Testing on AC-GAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
        filename = 'ac_gan'
    elif test_adasyn:
        print('Testing on ADASYN')
        synthetic_test_data = pickle.load(open('synthetic_test_data_adasyn', 'rb'))
        filename = 'adasyn'

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_test = test_data['label']

    # Drop label column
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_adasyn = scaler.fit_transform(X=X_adasyn)
    X_test = scaler.transform(X=X_test)

    y_adasyn = y_adasyn.astype('int')
    y_test = y_test.astype('int')

    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_adasyn, X_test, y_adasyn, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('./bot_evolution/multi_class/train_on_augmented/train_on_augmented_adasyn_and_test_on_augmented_' + filename + '_data.csv')


def train_on_cgan_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=False):
    print('\n ~~~~~~~~~~~ Train on Augmented and Test on Augmented Data ~~~~~~~~~~~~~\n')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    synthetic_train_data = pickle.load(
        open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))

    train_data = synthetic_train_data.append(original_train_data)
    train_data = train_data.sample(frac=1)

    y_train = train_data['label']
    X_train = train_data.drop(['label'], axis=1)

    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if test_cgan:
        print('Testing on CGAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
        filename = 'cgan'
    elif test_ac_gan:
        print('Testing on AC-GAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
        filename = 'ac_gan'
    elif test_adasyn:
        print('Testing on ADASYN')
        synthetic_test_data = pickle.load(open('synthetic_test_data_adasyn', 'rb'))
        filename = 'adasyn'

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_test = test_data['label']

    # Drop label column
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('./bot_evolution/multi_class/train_on_augmented/train_on_augmented_cgan_and_test_on_augmented_' + filename + '_data.csv')


def train_on_ac_gan_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=False):
    print('\n ~~~~~~~~~~~ Train on Augmented and Test on Augmented Data ~~~~~~~~~~~~~\n')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    synthetic_train_data = pickle.load(
        open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

    train_data = synthetic_train_data.append(original_train_data)
    train_data = train_data.sample(frac=1)

    y_train = train_data['label']
    X_train = train_data.drop(['label'], axis=1)

    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if test_cgan:
        print('Testing on CGAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
        filename = 'cgan'
    elif test_ac_gan:
        print('Testing on AC-GAN')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
        filename = 'ac_gan'
    elif test_adasyn:
        print('Testing on ADASYN')
        synthetic_test_data = pickle.load(open('synthetic_test_data_adasyn', 'rb'))
        filename = 'adasyn'

    test_data = synthetic_test_data.append(original_test_data)
    test_data = test_data.sample(frac=1)

    y_test = test_data['label']

    # Drop label column
    test_data = test_data.drop(['label'], axis=1)

    # split the labeled tweets into train and test set
    X_test = test_data

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')


    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('./bot_evolution/multi_class/train_on_augmented/train_on_augmented_ac_gan_and_test_on_augmented_' + filename + '_data.csv')


def test_on_augmented_mixed_data(train_original=False, train_cgan=False, train_ac_gan=False, train_both_gans=False):
    # Mixed test data are composed of: Original + CGAN + AC-GAN data
    print('\n ~~~~~~~~~~~ Test on Augmented Mixed Data from CGAN and AC-GAN ~~~~~~~~~~~~~\n')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))

    mixed_synthetic_test_data = pickle.load(open('../data/synthetic_data/mixed_synthetic_test_data', 'rb'))
    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    mixed_augmented_test_data = mixed_synthetic_test_data.append(original_test_data)

    if train_original:
        print('Training on original')
        filename = 'original'
        train_data = original_train_data
    elif train_cgan:
        print('Training on CGAN')
        synthetic_train_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))
        filename = 'cgan'
        train_data = synthetic_train_data.append(original_train_data)
        train_data = train_data.sample(frac=1)
    elif train_ac_gan:
        print('Training on AC-GAN')
        synthetic_train_data = pickle.load(open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))
        filename = 'ac_gan'
        train_data = synthetic_train_data.append(original_train_data)
        train_data = train_data.sample(frac=1)
    elif train_both_gans:
        print('Training on mixed augmented data from CGAN and AC-GAN')
        synthetic_mixed_train_data = pickle.load(open('../data/synthetic_data/mixed_synthetic_train_data', 'rb'))
        filename = 'with_mixed_train'
        train_data = synthetic_mixed_train_data.append(original_train_data)
        train_data = train_data.sample(frac=1)

    y_train = train_data['label']
    X_train = train_data.drop(['label'], axis=1)

    y_test = mixed_augmented_test_data['label']

    # Drop label column
    X_test = mixed_augmented_test_data.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')


    acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=False)

    df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
    df.to_csv('./bot_evolution/multi_class/test_on_mixed_augmented/test_on_mixed_augmented_' + filename + '_data.csv')


# Test on Original Data
# train_and_test_on_augmented_data(train_ac_gan=True, test_ac_gan=False)
#train_on_original_and_test_on_augmented_data(cgan=True, ac_gan=False, adasyn=False)
#train_on_original_and_test_on_augmented_data(cgan=False, ac_gan=True, adasyn=False)
#train_on_original_and_test_on_augmented_data(cgan=False, ac_gan=False, adasyn=True)

#train_on_adasyn_and_test_on_augmented_data(test_adasyn=False, test_cgan=True, test_ac_gan=False)
#train_on_adasyn_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=True)
#train_on_adasyn_and_test_on_augmented_data(test_adasyn=True, test_cgan=False, test_ac_gan=False)

#train_on_cgan_and_test_on_augmented_data(test_adasyn=False, test_cgan=True, test_ac_gan=False)
#train_on_cgan_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=True)
#train_on_cgan_and_test_on_augmented_data(test_adasyn=True, test_cgan=False, test_ac_gan=False)


#train_on_ac_gan_and_test_on_augmented_data(test_adasyn=False, test_cgan=True, test_ac_gan=False)
#train_on_ac_gan_and_test_on_augmented_data(test_adasyn=False, test_cgan=False, test_ac_gan=True)
#train_on_ac_gan_and_test_on_augmented_data(test_adasyn=True, test_cgan=False, test_ac_gan=False)

test_on_augmented_mixed_data(train_original=True, train_cgan=False, train_ac_gan=False)
test_on_augmented_mixed_data(train_original=False, train_cgan=True, train_ac_gan=False)
test_on_augmented_mixed_data(train_original=False, train_cgan=False, train_ac_gan=True)
test_on_augmented_mixed_data(train_original=False, train_cgan=False, train_ac_gan=False, train_both_gans=True)
