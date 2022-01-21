import pickle
from pprint import pprint

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from classifiers import evaluation_metrics


def custom_Parameter_Search(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=14, n_jobs=4, max_features=0.33, min_samples_leaf=1, max_depth=None)
    parameter_grid = {
        'criterion': ['gini', 'entropy'],
        #'max_depth': [None, 1],
        #'max_features': [0.33, 1.0],
        #'min_samples_leaf': [1, 5],
        'min_samples_split': [2, 5, 10],
        'n_estimators': list(range(100, 420, 20))
    }

    pprint(parameter_grid)
    previous_g_mean = 0
    i = 0
    for g in ParameterGrid(parameter_grid):
        i += 1
        print(i)
        rf.set_params(**g)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        g_mean = geometric_mean_score(y_test, y_pred, average='macro')

        # save rf parameters if current params give better g_mean than previous
        if g_mean > previous_g_mean:
            previous_g_mean = g_mean
            best_grid = g
            print('\nBest parameters updated...')
            print("G-Mean {:.10f}".format(g_mean))
            print("F1-score {:.10f}".format(f1_score(y_test, y_pred, average='macro')))
            pprint(best_grid)

    pprint(best_grid)

    return best_grid


def Grid_Search(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=14, n_jobs=4)
    parameter_grid = {
        'max_depth': [None, 1, 5, 15, 20],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'n_estimators': list(range(100, 300, 50))
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=parameter_grid,
                               cv=3, n_jobs=-1, verbose=3)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    pprint(best_params)
    y_pred = best_model.predict(X_test)
    evaluation_metrics(best_model, y_test, y_pred)

    return best_params, best_model


def prepare_data():
    original_train_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    original_test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))

    synthetic_data = pickle.load(
        open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))

    augmented_df = original_train_data.append(synthetic_data)
    augmented_df = augmented_df.sample(frac=1)

    y_train = augmented_df['label']
    X_train = augmented_df.drop(['label'], axis=1)

    y_test = original_test_data['label']
    X_test = original_test_data.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    best_params = custom_Parameter_Search(X_train, X_test, y_train, y_test)
    #best_params, best_model = Grid_Search(X_train, X_test, y_train, y_test)

    return best_params


prepare_data()