"""
Import necessary libraries to create a generative adversarial network
The code is developed using the PyTorch library
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sdv.evaluation import evaluate
from sdv.metrics.tabular import KSTest
from statistics import mean
import helper_functions.transform_booleans as transform_bool
import warnings
from sklearn import metrics
from imblearn.metrics import geometric_mean_score

warnings.filterwarnings('ignore')

"""
This model uses an Auxiliary Classifier (AC) to enable GAN perform both multiclass bot detection
and bot synthetic data generation.

Network Architectures
The following are the discriminator and generator architectures
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_classes = 6
        self.num_features = 310
        self.prob = 0.5

        self.model = nn.Sequential(
            nn.Linear(self.num_features, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.prob),
            nn.Linear(400, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.prob),
            nn.Linear(1000, 2000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.prob)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2000, 1),
            nn.Sigmoid()
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(2000, self.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(-1, self.num_features)
        x = self.model(x)
        source_out = self.discriminator(x)
        class_out = self.aux_classifier(x)
        return source_out.squeeze(), class_out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_classes = 6
        self.num_features = 310
        self.noise = 128
        # embedding layer of the class labels (num_of_classes * encoding_size of each word)
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.noise + self.num_classes, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, self.num_features),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), self.noise)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, 1, 1, self.num_features)


def prepare_data(df=pickle.load(open('../data/original_data/train_multiclass_data', 'rb')), batch_size=256):
    # df = df.sample(n=1000)

    # print(df['label'].value_counts())
    y = df['label']

    # Drop label column
    df = df.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(X=df)

    # Store scaler for later use
    scaler_filename = "ac_gan/scaler.save"
    joblib.dump(scaler, scaler_filename)

    # Transform dataframe into pytorch Tensor
    train = TensorDataset(torch.Tensor(df_scaled), torch.Tensor(np.array(y)))
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    return train_loader, df, pd.DataFrame(df_scaled)


def train_gan(epochs=100):
    """
    Determine if any GPUs are available
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Hyperparameter settings
    """
    G_lr = 0.0002
    D_lr = 0.0002
    bs = 512
    adversarial_loss = nn.BCELoss().to(device)
    auxiliary_loss = nn.CrossEntropyLoss().to(device)
    num_of_classes = 6

    # Model
    G = Generator().to(device)
    D = Discriminator().to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.999))

    # Load our data
    train_loader, _, _ = prepare_data(batch_size=bs)

    """
    Network training procedure
    Every step both the loss for disciminator and generator is updated
    Discriminator aims to classify reals and fakes
    Generator aims to generate bot accounts as realistic as possible
    """
    mean_D_loss = []
    mean_G_loss = []
    mean_D_acc = []
    mean_Gmean = []
    for epoch in range(epochs):
        epoch_D_loss = []
        epoch_G_loss = []
        epoch_D_acc = []
        epoch_acc = []
        epoch_Gmean = []
        epoch_precision = []
        # labels = train_loader[1]
        for idx, train_data in enumerate(train_loader):
            idx += 1

            # ---------------------
            #  Train Generator
            # ---------------------

            # Make a batch of fake samples using Generator
            # Feed fake samples to Discriminator, compute reverse loss and use it to update the Generator

            # Real inputs are actual samples from the original dataset
            # Fake inputs are from the generator
            real_inputs = train_data[0].to(device)
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            fake_class_labels = torch.randint(0, num_of_classes, (real_inputs.shape[0],)).to(device)
            fake_inputs = G(noise, fake_class_labels)
            fake_outputs, pred_label = D(fake_inputs)
            fake_outputs = fake_outputs.view(-1, 1)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)

            # Generator Loss
            G_loss = 0.5 * (
                    adversarial_loss(fake_outputs, fake_targets) + auxiliary_loss(pred_label, fake_class_labels))

            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            ### Loss for real samples

            # Fetch a batch of real samples from training data
            # Feed real samples to Discriminator
            class_labels = train_data[1].to(torch.int64).to(device)
            real_outputs, real_pred = D(real_inputs)
            real_outputs = real_outputs.view(-1, 1)
            real_targets = torch.ones(real_inputs.shape[0], 1).to(device)

            D_real_loss = (adversarial_loss(real_outputs, real_targets) + auxiliary_loss(real_pred, class_labels)) / 2

            ### Loss for fake samples

            # Make a batch of fake samples using Generator and then feed fake samples to Discriminator
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            fake_class_labels = torch.randint(0, num_of_classes, (real_inputs.shape[0],)).to(device)
            fake_inputs = G(noise, fake_class_labels)

            fake_outputs, fake_pred = D(fake_inputs)
            fake_outputs = fake_outputs.view(-1, 1)
            fake_targets = torch.zeros(fake_inputs.shape[0], 1).to(device)

            D_fake_loss = (adversarial_loss(fake_outputs, fake_targets) + auxiliary_loss(fake_pred,
                                                                                         fake_class_labels)) / 2

            # Discriminator total loss
            D_loss = (D_real_loss + D_fake_loss) / 2

            # Calculate Discriminator accuracy
            pred = np.concatenate([real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0)
            ground_truth = np.concatenate([class_labels.data.cpu().numpy(), fake_class_labels.data.cpu().numpy()],
                                          axis=0)

            D_predictions = np.argmax(pred, axis=1)

            D_accuracy = np.mean(D_predictions == ground_truth)

            # print("==============================")
            acc = metrics.accuracy_score(ground_truth, D_predictions)
            # print("Precision {:.5f}".format(metrics.precision_score(ground_truth, D_predictions, average='macro')))
            # print("F1-score {:.5f}".format(metrics.f1_score(ground_truth, D_predictions, average='macro')))
            # print("Recall-score {:.5f}".format(metrics.recall_score(ground_truth, D_predictions, average='macro')))
            # print("G-Mean {:.5f}".format(geometric_mean_score(ground_truth, D_predictions, correction=0.0001)))
            # print("==============================\n")
            Gmean = geometric_mean_score(ground_truth, D_predictions, correction=0.0001)
            precision = metrics.precision_score(ground_truth, D_predictions, average='macro')
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            if idx % 100 == 0 or idx == len(train_loader):
                epoch_D_loss.append(D_loss.item())
                epoch_G_loss.append(G_loss.item())
                epoch_D_acc.append(D_accuracy.item())
                epoch_Gmean.append(Gmean)
                epoch_acc.append(acc)
                epoch_precision.append(precision)

        if epoch > 290:
            torch.save(D, 'ac_gan/AC_GAN_Discriminator_save' + str(epoch) + '.pth')
        print(
            "[Epoch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, epochs, mean(epoch_D_loss), 100 * mean(epoch_D_acc), mean(epoch_G_loss)))
        print("~~~~~~~~~~~~~~ Discriminator Performance on Mixed Data ~~~~~~~~~~~~~~~~")
        print("G-Mean {:.5f}".format(mean(epoch_Gmean)))
        print("Precision {:.5f}".format(mean(epoch_precision)))
        print("==============================\n")

        # print('Epoch {} -- Discriminator mean Accuracy: {:.5f}'.format(epoch, mean(epoch_D_acc)))
        # print('Epoch {} -- Discriminator mean loss: {:.5f}'.format(epoch, mean(epoch_D_loss)))
        # print('Epoch {} -- Generator mean loss: {:.5f}'.format(epoch, mean(epoch_G_loss)))
        mean_D_loss.append(mean(epoch_D_loss))
        mean_G_loss.append(mean(epoch_G_loss))
        mean_D_acc.append(mean(epoch_D_acc))
        mean_Gmean.append(mean(epoch_Gmean))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(mean_D_loss, color='blue', label='Discriminator loss')
    plt.plot(mean_G_loss, color='red', label='Generator loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('AC-GAN Loss')
    plt.legend()
    plt.savefig('ac_gan/ac_gan_loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(mean_D_acc, color='blue', label='Discriminator accuracy')
    plt.plot(mean_Gmean, color='red', label='Discriminator G-Mean')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('AC-GAN Accuracy')
    plt.savefig('ac_gan/ac_gan_accuracy.png')
    plt.show()

    torch.save(G, 'ac_gan/AC_GAN_Generator_save.pth')
    torch.save(D, 'ac_gan/AC_GAN_Discriminator_save.pth')
    print('Generator and Discriminator saved.')


"""
A function that loads a trained Generator model and uses it to create synthetic samples
"""


def generate_synthetic_samples(num_of_samples=100, num_of_features=310, label=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load initial data
    _, real_data, _ = prepare_data()

    generator = torch.load('ac_gan/AC_GAN_Generator_save.pth')
    generator.eval()
    # Generate points in the latent space
    noise = (torch.rand(num_of_samples, 128) - 0.5) / 0.5
    noise = noise.to(device)

    # Create class labels
    class_labels = torch.randint(label, label + 1, (num_of_samples,)).to(device)

    # Pass latent points and class labels through our Generator to produce synthetic samples
    synthetic_samples = generator(noise, class_labels)

    # Transform pytorch tensor to numpy array
    synthetic_samples = synthetic_samples.cpu().detach().numpy()
    synthetic_samples = synthetic_samples.reshape(num_of_samples, num_of_features)
    class_labels = class_labels.cpu().detach().numpy()
    class_labels = class_labels.reshape(num_of_samples, 1)

    # Load saved min_max_scaler for inverse scaling transformation of the generated data
    scaler = joblib.load("ac_gan/scaler.save")
    synthetic_data = scaler.inverse_transform(synthetic_samples)
    synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

    synthetic_samples = synthetic_data.copy(deep=True)
    # Insert column containing labels
    synthetic_data.insert(loc=310, column='label', value=class_labels, allow_duplicates=True)
    # Round values to closest integer for columns that should be boolean
    synthetic_data = transform_bool.transform(synthetic_data)
    # Map booleans to 1 and 0.
    synthetic_data = synthetic_data.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    # pickle.dump(synthetic_data, open('../data/synthetic_data/ac_gan/synthetic_data' + str(num_of_samples), 'wb'))

    return synthetic_data, real_data


def generate_samples_to_reach_30K_per_class():
    ## For each class, generate that many synthetic samples to reach 30000 so that we have a balanced dataset.
    """
    Label Distribution:
    0    24403
    3    13283
    1     9333
    5     4703
    4      958
    2      408
    """
    ## For each class, generate that many samples to reach 30000 samples
    synthetic_data0, _ = generate_synthetic_samples(num_of_samples=30000 - 24403, label=0)
    synthetic_data1, _ = generate_synthetic_samples(num_of_samples=30000 - 9333, label=1)
    synthetic_data2, _ = generate_synthetic_samples(num_of_samples=30000 - 408, label=2)
    synthetic_data3, _ = generate_synthetic_samples(num_of_samples=30000 - 13283, label=3)
    synthetic_data4, _ = generate_synthetic_samples(num_of_samples=30000 - 958, label=4)
    synthetic_data5, _ = generate_synthetic_samples(num_of_samples=30000 - 4703, label=5)

    # List of above dataframes
    pdList = [synthetic_data0, synthetic_data1, synthetic_data2, synthetic_data3, synthetic_data4, synthetic_data5]
    final_df = pd.concat(pdList)

    # Shuffle the dataframe
    final_df = final_df.sample(frac=1)

    pickle.dump(final_df,
                open('../data/synthetic_data/ac_gan/synthetic_data_30K_per_class', 'wb'))
    return final_df


def generate_2to1_synthetic_samples():
    ## For each class, generate 2:1 synthetic samples, except humans.
    """
    Label Distribution:
    0    24403
    1     9333
    2      408
    3    13283
    4      958
    5     4703
    """
    ## For each class, generate that many samples to reach 30000 samples
    synthetic_data1, _ = generate_synthetic_samples(num_of_samples=2*9333, label=1)
    synthetic_data2, _ = generate_synthetic_samples(num_of_samples=2*408, label=2)
    synthetic_data3, _ = generate_synthetic_samples(num_of_samples=2*13283, label=3)
    synthetic_data4, _ = generate_synthetic_samples(num_of_samples=2*958, label=4)
    synthetic_data5, _ = generate_synthetic_samples(num_of_samples=2*4703, label=5)

    # List of above dataframes
    pdList = [synthetic_data1, synthetic_data2, synthetic_data3, synthetic_data4, synthetic_data5]
    final_df = pd.concat(pdList)

    # Shuffle the dataframe
    final_df = final_df.sample(frac=1)

    pickle.dump(final_df,
                open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'wb'))
    return final_df


def generate_test_synthetic_data(equal_to_adasyn=False):
    if equal_to_adasyn:
        ## For each class, generate 2200 samples for testing
        synthetic_data0, _ = generate_synthetic_samples(num_of_samples=6633, label=0)
        synthetic_data1, _ = generate_synthetic_samples(num_of_samples=6657, label=1)
        synthetic_data2, _ = generate_synthetic_samples(num_of_samples=6318, label=2)
        synthetic_data3, _ = generate_synthetic_samples(num_of_samples=6328, label=3)
        synthetic_data4, _ = generate_synthetic_samples(num_of_samples=6224, label=4)
        synthetic_data5, _ = generate_synthetic_samples(num_of_samples=6437, label=5)
        filename = 'synthetic_test_data_custom'
    else:

        """
        Original Test Data Label Distribution:
        0    6032
        3    3431
        1    2306
        5    1177
        4     239
        2     88
        """
        ## For each class, generate that many samples to reach 30000 samples
        synthetic_data0, _ = generate_synthetic_samples(num_of_samples=6032, label=0)
        synthetic_data1, _ = generate_synthetic_samples(num_of_samples=2306, label=1)
        synthetic_data2, _ = generate_synthetic_samples(num_of_samples=88, label=2)
        synthetic_data3, _ = generate_synthetic_samples(num_of_samples=3431, label=3)
        synthetic_data4, _ = generate_synthetic_samples(num_of_samples=239, label=4)
        synthetic_data5, _ = generate_synthetic_samples(num_of_samples=1177, label=5)

        filename = 'synthetic_test_data'

    # List of above dataframes
    pdList = [synthetic_data0, synthetic_data1, synthetic_data2, synthetic_data3, synthetic_data4, synthetic_data5]
    final_df = pd.concat(pdList)

    # Shuffle the dataframe
    final_df = final_df.sample(frac=1)

    pickle.dump(final_df, open('../data/synthetic_data/ac_gan/' + filename, 'wb'))
    return final_df


def evaluate_synthetic_data():
    real_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))
    real_data = real_data.drop(['label'], axis=1)

    test_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))
    test_data = test_data.drop(['label'], axis=1)

    print('\n~~~~~~~~~~~~~~ Synthetic Data Evaluation ~~~~~~~~~~~~~~')

    print('\n~~~~~~~~~~~~~~ Evaluating method of creating that many samples to reach 30K per class ~~~~~~~~~~~~~~')
    synthetic_data = pickle.load(
        open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

    synthetic_data = synthetic_data.drop(['label'], axis=1)

    ks = KSTest.compute(synthetic_data, real_data)
    ks_test_data = KSTest.compute(synthetic_data, test_data)
    print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))
    print('Inverted Kolmogorov-Smirnov D statistic on Test Data: {}'.format(ks_test_data))

    kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
    print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))

    print('\n~~~~~~~~~~~~~~ Evaluating method of creating two to one synthetic samples  per class ~~~~~~~~~~~~~~')
    synthetic_data = pickle.load(
        open('../data/synthetic_data/ac_gan/synthetic_data_2_to_1', 'rb'))

    synthetic_data = synthetic_data.drop(['label'], axis=1)

    ks = KSTest.compute(synthetic_data, real_data)
    ks_test_data = KSTest.compute(synthetic_data, test_data)
    print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))
    print('Inverted Kolmogorov-Smirnov D statistic on Test Data: {}'.format(ks_test_data))

    kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
    print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))


def predict_bot_class(synthetic_data, d=299):
    scaler = joblib.load("ac_gan/scaler.save")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    Use trained Discriminator to predict the bot type of unseen original
    and synthetic data and evaluate its performance.
    """

    ############################  Classify only Original Data  ############################
    test_original_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))
    # Get the label from our data
    ground_truth = test_original_data['label']

    test_orig_data = test_original_data.drop('label', axis=1)
    test_orig_data = scaler.transform(test_orig_data)
    test_orig_data = torch.tensor(test_orig_data, dtype=torch.float32).to(device)

    disc = str(d)
    print('Evaluating with Discriminator of epoch:', d)
    discriminator = torch.load('ac_gan/AC_GAN_Discriminator_save.pth')
    discriminator.eval()
    _, predictions = discriminator(test_orig_data)
    predictions = predictions.cpu().detach().numpy()
    D_predictions = np.argmax(predictions, axis=1)

    print("~~~~~~~~~~~~~~ Discriminator Performance on Original Data ~~~~~~~~~~~~~~~~")
    print("Accuracy {:.5f}".format(metrics.accuracy_score(ground_truth, D_predictions)))
    print("Precision {:.5f}".format(metrics.precision_score(ground_truth, D_predictions, average='macro')))
    print("F1-score {:.5f}".format(metrics.f1_score(ground_truth, D_predictions, average='macro')))
    print("Recall-score {:.5f}".format(metrics.recall_score(ground_truth, D_predictions, average='macro')))
    print("G-Mean {:.5f}".format(geometric_mean_score(ground_truth, D_predictions, average='macro')))
    print("==============================\n")

    ############################  Classify only Synthetic Data  ############################
    y_test = synthetic_data['label']
    X_train = synthetic_data.drop('label', axis=1)
    X_train = scaler.transform(X_train)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

    _, predictions = discriminator(X_train)
    D_predictions = np.argmax(predictions.cpu().detach().numpy(), axis=1)

    print("~~~~~~~~~~~~~~ Discriminator Performance on Synthetic Data ~~~~~~~~~~~~~~~~")
    print("Accuracy {:.5f}".format(metrics.accuracy_score(y_test, D_predictions)))
    print("Precision {:.5f}".format(metrics.precision_score(y_test, D_predictions, average='macro')))
    print("F1-score {:.5f}".format(metrics.f1_score(y_test, D_predictions, average='macro')))
    print("Recall-score {:.5f}".format(metrics.recall_score(y_test, D_predictions, average='macro')))
    print("G-Mean {:.5f}".format(geometric_mean_score(y_test, D_predictions, average='macro')))
    print("==============================\n")

    ############################  Classify on Mixed Data  ############################

    mixed_data = test_original_data.append(synthetic_data)
    mixed_data = mixed_data.sample(frac=1)

    ground_truth = mixed_data['label']
    mixed_data = mixed_data.drop('label', axis=1)
    mixed_data = scaler.transform(mixed_data)
    mixed_data = torch.tensor(mixed_data, dtype=torch.float32).to(device)

    _, predictions = discriminator(mixed_data)
    D_predictions = np.argmax(predictions.cpu().detach().numpy(), axis=1)

    print("~~~~~~~~~~~~~~ Discriminator Performance on Mixed Data ~~~~~~~~~~~~~~~~")
    print("Accuracy {:.5f}".format(metrics.accuracy_score(ground_truth, D_predictions)))
    print("Precision {:.5f}".format(metrics.precision_score(ground_truth, D_predictions, average='macro')))
    print("F1-score {:.5f}".format(metrics.f1_score(ground_truth, D_predictions, average='macro')))
    print("Recall-score {:.5f}".format(metrics.recall_score(ground_truth, D_predictions, average='macro')))
    print("G-Mean {:.5f}".format(geometric_mean_score(ground_truth, D_predictions, average='macro')))
    print("==============================\n")


def ac_gan_classification(test_ac_gan=True, test_cgan=False, d=299):
    if test_ac_gan:
        print('------- Classification on AC-GAN test data -------')
        synthetic_test_data = pickle.load(open('../data/synthetic_data/ac_gan/synthetic_test_data_custom', 'rb'))
        test_filename = 'ac_gan'
    elif test_cgan:
        print('------- Classification on Conditional GAN test data -------')
        synthetic_test_data = pickle.load(
            open('../data/synthetic_data/conditional_gan_multiclass/synthetic_test_data_custom', 'rb'))
        test_filename = 'cgan'
    else:
        print('------- Classification on mixed data from CGAN and AC-GAN -------')
        synthetic_test_data = pickle.load(open('../data/synthetic_data/mixed_synthetic_test_data', 'rb'))

    predict_bot_class(synthetic_test_data, d=d)


#train_gan(epochs=300)

#generate_samples_to_reach_30K_per_class()
#generate_2to1_synthetic_samples()

#print('\n~~~~~~~~~~~~~~ Evaluating method of creating 30K new samples for each class ~~~~~~~~~~~~~~')
#evaluate_synthetic_data()

#generate_test_synthetic_data(equal_to_adasyn=True)
ac_gan_classification(test_ac_gan=False, test_cgan=True)

