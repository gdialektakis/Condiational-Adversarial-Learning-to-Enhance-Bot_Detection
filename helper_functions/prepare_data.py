import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def filter_data_by_dataset():
    data = pickle.load(open('../data/original_data/final_data_no_rts_v2', 'rb'))
    usermap = pickle.load(open('../data/original_data/user_dataset_mapping', 'rb'))

    df = pd.DataFrame(data)
    df['dataset'] = df['user_id'].map(usermap)

    caverlee_df = df[df['dataset'] == 'Caverlee']
    varol_df = df[df['dataset'] == 'Varol']
    cresci_df = df[df['dataset'] == 'Cresci_Stock']

    gilani_df = df[df['dataset'] == 'Gilani']

    cresci_df = cresci_df.drop(['dataset'], axis=1)
    varol_df = varol_df.drop(['dataset'], axis=1)
    caverlee_df = caverlee_df.drop(['dataset'], axis=1)

    gilani_df = gilani_df.drop(['dataset'], axis=1)

    #pickle.dump(caverlee_df, open('../binary_data/old_data', 'wb'))
    #pickle.dump(varol_df, open('../binary_data/new_data', 'wb'))
    #pickle.dump(cresci_df, open('../binary_data/new_data2', 'wb'))
    pickle.dump(gilani_df, open('../binary_data/new_data3', 'wb'))
    #df_varol = pickle.load(open('../binary_data/new_data', 'rb'))
    #df_caverlee = pickle.load(open('../binary_data/old_data', 'rb'))


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
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    pickle.dump(test_set, open('../data/test_binary_data' + filename, 'wb'))
    pickle.dump(train_set, open('../data/train_binary_data' + filename, 'wb'))


def prepare_old_data(data=pickle.load(open('../binary_data/old_data', 'rb')), bots=True):
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
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    pickle.dump(test_set, open('../binary_data/test_old_data_' + filename, 'wb'))
    pickle.dump(train_set, open('../binary_data/train_old_data_' + filename, 'wb'))


def prepare_new_data(data=pickle.load(open('../binary_data/new_data3', 'rb'))):
    df = pd.DataFrame(data)
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    # Convert features that are boolean to integers
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    # Drop unwanted columns
    df = df.drop(['user_name', 'user_screen_name', 'user_id', 'time'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    pickle.dump(df, open('../binary_data/new_data3', 'wb'))


def merge_dataframes(old_data=True):
    # This function merges bots and humans into a single dataframe.

    if old_data:
        bots_train = pickle.load(open('../binary_data/train_old_data_bots', 'rb'))
        humans_train = pickle.load(open('../binary_data/train_old_data_humans', 'rb'))

        train_data = bots_train.append(humans_train)
        train_data = train_data.sample(frac=1)
        pickle.dump(train_data, open('../binary_data/train_old_data', 'wb'))

        bots_test = pickle.load(open('../binary_data/test_old_data_bots', 'rb'))
        humans_test = pickle.load(open('../binary_data/test_old_data_humans', 'rb'))

        test_data = bots_test.append(humans_test)
        test_data = test_data.sample(frac=1)
        pickle.dump(test_data, open('../binary_data/test_old_data', 'wb'))

    else:

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


#prepare_old_data(bots=True)
#prepare_old_data(bots=False)
#merge_dataframes(old_data=True)

#filter_data_by_dataset()
prepare_new_data()

