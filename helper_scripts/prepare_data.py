import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch


def prepare_multiclass_data(df=pickle.load(open('../data/multi_class_data', 'rb'))):
    # df = df.sample(n=1000)
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    # Keep 20% of the data for later testing
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    pickle.dump(test_set, open('../data/test_multiclass_data', 'wb'))
    pickle.dump(train_set, open('../data/train_multiclass_data', 'wb'))


def prepare_binary_data(data=pickle.load(open('../data/final_data_no_rts_v2', 'rb')), bots=True):
    df = pd.DataFrame(data)
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    # Convert features that are boolean to integers
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id'], axis=1)
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
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    pickle.dump(test_set, open('../data/test_binary_data_'+filename, 'wb'))
    pickle.dump(train_set, open('../data/train_binary_data_'+filename, 'wb'))


prepare_binary_data(bots=True)
prepare_binary_data(bots=False)
prepare_multiclass_data()