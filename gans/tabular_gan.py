import joblib
from ctgan import CTGANSynthesizer
from sdv.evaluation import evaluate
import pickle
import pandas as pd
import warnings
from sdv.metrics.tabular import KSTest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def prepare_data():
    # Load our data
    bot_data = pickle.load(open('../data/final_data_no_rts_v2', 'rb'))
    df = pd.DataFrame(bot_data)

    # convert labels from string to 0 and 1
    df['label'] = df['label'].map({'human': 0, 'bot': 1, 'cyborg': 1})
    # keep only bot accounts to train our GAN
    df = df[df['label'] == 1]
    #df = df.sample(n=1000)

    # Keep 20% of the data for later testing
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    pickle.dump(test_set, open('ctgan/ctgan_test_data', 'wb'))

    # Convert features that are boolean to integers
    df = train_set.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    y = df['label']

    # Drop unwanted columns
    df = train_set.drop(['user_name', 'user_screen_name', 'user_id', 'label'], axis=1)
    if 'max_appearance_of_punc_mark' in df.columns:
        df = df.drop(['max_appearance_of_punc_mark'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(X=df)

    # Store scaler for later use
    scaler_filename = "ctgan/ctgan_scaler.save"
    joblib.dump(scaler, scaler_filename)

    return df_scaled, df


def train_CTGAN(epochs=10):

    data, _ = prepare_data()

    # Initialize a CTGAN model and train it
    ctgan = CTGANSynthesizer(epochs=epochs, verbose=True, generator_dim=(500, 1000, 2000), discriminator_dim=(400, 500))
    print('Training starting')
    ctgan.fit(data)

    ctgan.save('ctgan/ctgan.pkl')


def generate_synthetic_samples(num_of_samples=100):
    saved_ctgan = CTGANSynthesizer.load('ctgan/ctgan.pkl')

    # Create synthetic data
    synthetic_samples = saved_ctgan.sample(num_of_samples)

    _, real_data = prepare_data()

    # Load saved min_max_scaler for inverse transformation of the generated data
    scaler = joblib.load("ctgan/ctgan_scaler.save")
    synthetic_data = scaler.inverse_transform(synthetic_samples)
    synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

    pickle.dump(synthetic_data, open('ctgan/ctgan_synthetic_data_' + str(num_of_samples), 'wb'))
    print(synthetic_data)
    return synthetic_data, real_data


#train_CTGAN(epochs=300)


synthetic_data, real_data = generate_synthetic_samples(num_of_samples=30000)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Data evaluation: {}'.format(KSTest.compute(synthetic_data, real_data)))