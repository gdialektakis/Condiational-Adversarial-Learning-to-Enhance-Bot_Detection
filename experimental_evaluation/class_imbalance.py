import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve
from imblearn.combine import SMOTEENN
from classifiers import run_classifiers, results_to_df, evaluation_metrics
from sklearn.metrics import classification_report
from roc_curves import draw_plots
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

    print(classification_report(y_test, y_pred, digits=4))

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


############################### CLASS IMBALANCE ########################################################

def train_and_test_on_original_data(adasyn=False, smote=False, fraction=1.0, rf=False):
    train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    train_data = train_data.sample(frac=fraction, random_state=14)

    frac = 'full'
    if fraction == 0.5:
        print('Train with 1/2 training samples')
        frac = 'limited'
    elif fraction == 0.25:
        print('Train with 1/4 training samples')
        frac = 'short'
    else:
        print('Train with full training set')

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

    # summarize class distribution
    # print("\nLabel distribution before ADASYN: {}".format(Counter(y_train)))
    if rf:
        # ADASYN
        if adasyn:
            print('\n~~~~~~~~ Training Random Forest with ADASYN ~~~~~~~~~~~~~~~')
            adasyn = ADASYN(random_state=42, n_jobs=-1)
            X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
            acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_adasyn, X_test, y_adasyn, y_test, print_ROC=True)
            df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
            df.to_csv('./class_imbalance/original/rf_train_on_original_' + frac + '_adasyn.csv')
        elif smote:
            print('\n~~~~~~~~ Training Random Forest with SMOTE ENN ~~~~~~~~~~~~~~~')
            smote = SMOTEENN(random_state=42, n_jobs=-1)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
            acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_smote, X_test, y_smote, y_test, print_ROC=True)
            df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
            df.to_csv('./class_imbalance/original/rf_train_on_original_' + frac + '_smote.csv')
        else:
            print('\n~~~~~~~~ Training Random Forest ~~~~~~~~~~~~~~~')
            acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=True)
            df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
            df.to_csv('./class_imbalance/original/rf_train_on_original_' + frac + '.csv')
    else:
        print('\n~~~~~~~~ Training all Classifiers ~~~~~~~~~~~~~~~')
        acc, prec, f1, rec, g_m = run_classifiers(X_train, X_test, y_train, y_test)
        df = results_to_df(acc, prec, f1, rec, g_m, auc_score=0, pr_score=0, rf=False)
        df.to_csv('./class_imbalance/original/all_classifiers_train_on_original_' + frac + '.csv')


def train_on_augmented_and_test_on_original_data(ac_gan=False, two_to_1=False, specific_classes=False, fraction=1.0, rf=False):
    print('\n ~~~~~~~~~~~~~~~ Train with Augmented Data and Test on Original ~~~~~~~~~~~~~~~~')

    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    original_train_data = original_train_data.sample(frac=fraction, random_state=14)
    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    if two_to_1:
        print('\n---------------- Training with 2:1 new synthetic samples for each class  -------------------')
        method = '2to1'
    else:
        print('\n---------------- Training with 30K samples for each class  -------------------')
        method = '30K'

    if ac_gan:
        if two_to_1:
            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))
        else:
            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_30K_per_class', 'rb'))
        filename = 'ac_gan_'
    else:
        if two_to_1:
            synthetic_data = pickle.load(
                    open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))

        else:
            synthetic_data = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_30K_per_class', 'rb'))

        filename = 'cgan_'

    augmented_df = original_train_data.append(synthetic_data)
    augmented_df = augmented_df.sample(frac=1)

    y = augmented_df['label']
    print(augmented_df['label'].value_counts())

    # Drop unwanted columns
    augmented_df = augmented_df.drop(['label'], axis=1)

    X_train = augmented_df
    y_train = y

    y_test = original_test_data['label']

    # Drop unwanted columns
    original_test_data = original_test_data.drop(['label'], axis=1)

    X_test = original_test_data
    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    if fraction == 1.0:
        frac = '_full'
    elif fraction == 0.5:
        frac = '_limited'
    else:
        frac = '_short'

    if rf:
        print('\n~~~~~~~~ Training Random Forest ~~~~~~~~~~~~~~~')
        acc, prec, f1, rec, g_m, auc_score, pr_score = run_RF(X_train, X_test, y_train, y_test, print_ROC=True)
        df = results_to_df(acc, prec, f1, rec, g_m, auc_score, pr_score, rf=True)
        df.to_csv('./class_imbalance/augmented/rf_train_on_augmented_' + filename + method + frac + '.csv')
    else:
        print('\n~~~~~~~~ Training All Classifiers ~~~~~~~~~~~~~~~')
        acc, prec, f1, rec, g_m = run_classifiers(X_train, X_test, y_train, y_test)
        df = results_to_df(acc, prec, f1, rec, g_m, auc_score=0, pr_score=0, rf=False)
        df.to_csv('./class_imbalance/augmented/all_classifiers_train_on_augmented_' + filename + frac + '.csv')


# Test on Original Data
#train_and_test_on_original_data(adasyn=False, smote=False, fraction=1.0, rf=True)
train_on_augmented_and_test_on_original_data(ac_gan=False, two_to_1=True, fraction=0.25, rf=True)
#draw_plots(fraction=0.5)
