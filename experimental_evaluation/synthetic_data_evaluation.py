import pickle
import joblib
import pandas as pd
from sdv.evaluation import evaluate
from sdv.metrics.tabular import KSTest
from sdv.metrics.tabular import GMLogLikelihood


def evaluate_synthetic_data(cgan=True, ac_gan=False):
    real_data_with_label = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    real_data = real_data_with_label.drop(['label'], axis=1)

    print('\n~~~~~~~~~~~~~~ Synthetic Data Evaluation ~~~~~~~~~~~~~~')

    print('\n~~~~~~~~~~~~~~ Evaluating GAN Synthetic Data ~~~~~~~~~~~~~~')
    if cgan or ac_gan:
        if cgan:
            print("Evaluating CGAN Synthetic Data")
            synthetic_data_30K = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_30K_per_class', 'rb'))

            synthetic_data_30K = synthetic_data_30K.drop(['label'], axis=1)

            synthetic_data = pickle.load(
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))

            synthetic_data_2to1 = synthetic_data.drop(['label'], axis=1)
            # score = MulticlassDecisionTreeClassifier.compute(real_data_with_label, synthetic_data, target='label')
            # print('Efficacy Evaluation on CGAN data: {}'.format(score))
            bnl_score = GMLogLikelihood.compute(real_data.fillna(0), synthetic_data_2to1.fillna(0),
                                                n_components=(1, 10), retries=2, iterations=3)
            # bnl_score = GMLogLikelihood.normalize(bnl_score)
            print('BNLikelihood Evaluation on CGAN data: {}'.format(bnl_score))
        elif ac_gan:
            print("Evaluating AC-GAN Synthetic Data")
            synthetic_data_30K = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_30K_per_class', 'rb'))

            synthetic_data_30K = synthetic_data_30K.drop(['label'], axis=1)

            synthetic_data = pickle.load(
                open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

            synthetic_data_2to1 = synthetic_data.drop(['label'], axis=1)
            # score = MulticlassDecisionTreeClassifier.compute(real_data_with_label, synthetic_data, target='label')
            # print('Efficacy Evaluation on AC-GAN data: {}'.format(score))
            bnl_score = GMLogLikelihood.compute(real_data.fillna(0), synthetic_data_2to1.fillna(0),
                                                n_components=(1, 5), retries=2, iterations=2)
            # bnl_score = GMLogLikelihood.normalize(bnl_score)
            print('BNLikelihood Evaluation on CGAN data: {}'.format(eval))

        ks = KSTest.compute(synthetic_data_2to1, real_data)
        print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))

        kl_divergence = evaluate(synthetic_data_2to1, real_data, metrics=['ContinuousKLDivergence'])
        print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))

    else:
        print("Evaluating ADASYN Synthetic Data")
        scaler = joblib.load("scaler.save")
        synthetic_data_adasyn = pickle.load(open('synthetic_data_adasyn', 'rb'))
        synthetic_data = scaler.inverse_transform(synthetic_data_adasyn)
        synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

        ks = KSTest.compute(synthetic_data, real_data)
        print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))

        kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
        print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))

# evaluate_synthetic_data(cgan=False, ac_gan=True)
