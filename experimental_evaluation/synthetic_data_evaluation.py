import pickle

import pandas as pd
from sdv.evaluation import evaluate
from sdv.metrics.tabular import KSTest


def evaluate_synthetic_data(synthetic_data_adasyn, scaler, cgan=True, ac_gan=False):
    real_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    real_data = real_data.drop(['label'], axis=1)

    print('\n~~~~~~~~~~~~~~ Synthetic Data Evaluation ~~~~~~~~~~~~~~')

    print('\n~~~~~~~~~~~~~~ Evaluating ADASYN Synthetic Data ~~~~~~~~~~~~~~')
    if cgan or ac_gan:
        if cgan:
            print("Evaluating CGAN Synthetic Data")
            synthetic_data = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_30K_per_class', 'rb'))

            synthetic_data_30K = synthetic_data.drop(['label'], axis=1)

            synthetic_data = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))

            synthetic_data_2to1 = synthetic_data.drop(['label'], axis=1)
        elif ac_gan:
            print("Evaluating AC-GAN Synthetic Data")
            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_30K_per_class', 'rb'))

            synthetic_data_30K = synthetic_data.drop(['label'], axis=1)

            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

            synthetic_data_2to1 = synthetic_data.drop(['label'], axis=1)

        ks = KSTest.compute(synthetic_data_2to1, real_data)
        print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))

        kl_divergence = evaluate(synthetic_data_2to1, real_data, metrics=['ContinuousKLDivergence'])
        print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))

    else:
        print("Evaluating ADASYN Synthetic Data")
        synthetic_data = scaler.inverse_transform(synthetic_data_adasyn)
        synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

        ks = KSTest.compute(synthetic_data, real_data)
        print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))

        kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
        print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))
