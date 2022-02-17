import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch


def prepare_multiclass_data(df=pickle.load(open('../data/original_data/multi_class_data', 'rb'))):
    # df = df.sample(n=1000)
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    # Keep 20% of the data for later testing
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
    pickle.dump(test_set, open('../data/original_data/test_multiclass_data', 'wb'))
    pickle.dump(train_set, open('../data/original_data/train_multiclass_data', 'wb'))


def prepare_binary_data(data=pickle.load(open('../data/original_data/final_data_no_rts_v2', 'rb')), bots=True):
    df = pd.DataFrame(data)
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    # Convert features that are boolean to integers
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id', 'time'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    if bots:
        # keep only bot accounts to train our GAN
        df = df[df['label'] == 1]
        filename = 'bots'
    else:
        # keep only human accounts to train our GAN
        df = df[df['label'] == 0]
        filename = 'humans'

    # Keep 20% of the data for later testing
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
    pickle.dump(test_set, open('../data/test_binary_data' + filename, 'wb'))
    pickle.dump(train_set, open('../data/train_binary_data' + filename, 'wb'))


def merge_dataframes():
    # This function merges bots and humans into a single dataframe.
    bots_train = pickle.load(open('../data/original_data/train_binary_data_bots', 'rb'))
    humans_train = pickle.load(open('../data/original_data/train_binary_data_humans', 'rb'))

    train_data = bots_train.append(humans_train)
    train_data = train_data.sample(frac=1)
    pickle.dump(train_data, open('../data/original_data/train_binary_data', 'wb'))

    bots_test = pickle.load(open('../data/original_data/test_binary_data_bots', 'rb'))
    humans_test = pickle.load(open('../data/original_data/test_binary_data_humans', 'rb'))

    test_data = bots_test.append(humans_test)
    test_data = test_data.sample(frac=1)
    pickle.dump(test_data, open('../data/original_data/test_binary_data', 'wb'))


def create_mixed_augmented_test_dataset():
    cgan_test_data = pickle.load(
        open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
    ac_gan_test_data = pickle.load(
        open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))

    mixed_synthetic_data = cgan_test_data.append(ac_gan_test_data)
    mixed_synthetic_data = mixed_synthetic_data.sample(frac=1)

    pickle.dump(mixed_synthetic_data, open('../data/synthetic_data/mixed_synthetic_test_data', 'wb'))


def create_mixed_augmented_train_dataset():
    cgan_train_data = pickle.load(
        open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_2_to_1', 'rb'))
    ac_gan_train_data = pickle.load(
        open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

    mixed_synthetic_data = cgan_train_data.append(ac_gan_train_data)
    mixed_synthetic_data = mixed_synthetic_data.sample(frac=1)

    pickle.dump(mixed_synthetic_data, open('../data/synthetic_data/mixed_synthetic_train_data', 'wb'))


# prepare_binary_data(bots=True)
# prepare_binary_data(bots=False)
# merge_dataframes()
# prepare_multiclass_data()
create_mixed_augmented_test_dataset()
create_mixed_augmented_train_dataset()
