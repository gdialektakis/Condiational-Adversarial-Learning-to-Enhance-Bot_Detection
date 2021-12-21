"""
Import necessary libraries to create a generative adversarial network
The code is developed using the PyTorch library
"""
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sdv.evaluation import evaluate
from sdv.metrics.tabular import KSTest
from statistics import mean
import helper_scripts.transform_booleans as transform_bool
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

"""
Network Architectures
The following are the discriminator and generator architectures
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_classes = 6
        self.num_features = 310
        # embedding layer of the class labels (num_of_classes * encoding_size of each word)
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.num_features + self.num_classes, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(400, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), self.num_features)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


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

    #print(df['label'].value_counts())
    y = df['label']

    # Drop label column
    df = df.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(X=df)

    # Store scaler for later use
    scaler_filename = "conditional_gan_multi/scaler.save"
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
    loss = nn.BCELoss()
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
    D_acc = []
    for epoch in range(epochs):
        acc = []
        epoch_D_loss = []
        epoch_G_loss = []
        # labels = train_loader[1]
        for idx, train_data in enumerate(train_loader):
            idx += 1

            # Training the discriminator
            # Real inputs are actual samples from the original dataset
            # Fake inputs are from the generator
            # Real inputs should be classified as 1 and fake as 0

            # Fetch a batch of real samples from training data
            # Feed real samples to Discriminator
            real_inputs = train_data[0].to(device)
            class_labels = train_data[1].to(torch.int64).to(device)
            real_outputs = D(real_inputs, class_labels)
            real_labels = torch.ones(real_inputs.shape[0], 1).to(device)

            # Make a batch of fake samples using Generator
            # Feed fake samples to Discriminator
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            fake_class_labels = torch.randint(0, num_of_classes, (real_inputs.shape[0],)).to(device)
            fake_inputs = G(noise, fake_class_labels)

            fake_outputs = D(fake_inputs, fake_class_labels)
            fake_labels = torch.zeros(fake_inputs.shape[0], 1).to(device)

            # Combine the two loss values
            # use combined loss to update Discriminator
            outputs = torch.cat((real_outputs, fake_outputs), 0).view(-1, 1)
            targets = torch.cat((real_labels, fake_labels), 0)

            # Just for evaluation and monitoring purposes
            predictions = outputs.cpu().detach().numpy()
            predictions = np.round(predictions)
            labels = targets.cpu().detach().numpy()

            D_loss = loss(outputs, targets)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Training the Generator
            # For the Generator, the goal is to make the Discriminator believe everything is 1
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            # Make a batch of fake samples using Generator
            # Feed fake samples to Discriminator, compute reverse loss and use it to update the Generator
            fake_inputs = G(noise, fake_class_labels)
            fake_outputs = D(fake_inputs, fake_class_labels).view(-1, 1)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = loss(fake_outputs, fake_targets)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            if idx % 100 == 0 or idx == len(train_loader):
                epoch_D_loss.append(D_loss.item())
                epoch_G_loss.append(G_loss.item())
                D_accuracy = accuracy_score(labels, predictions)
                acc.append(D_accuracy)

        print('Epoch {} -- Discriminator mean Accuracy: {:.5f}'.format(epoch, mean(acc)))
        print('Epoch {} -- Discriminator mean loss: {:.5f}'.format(epoch, mean(epoch_D_loss)))
        print('Epoch {} -- Generator mean loss: {:.5f}'.format(epoch, mean(epoch_G_loss)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        mean_D_loss.append(mean(epoch_D_loss))
        mean_G_loss.append(mean(epoch_G_loss))
        D_acc.append(mean(acc))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(mean_D_loss, color='blue', label='Discriminator loss')
    plt.plot(mean_G_loss, color='red', label='Generator loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('conditional_gan_multi/cond_gan_loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(D_acc, color='blue', label='Discriminator accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('conditional_gan_multi/cond_discriminator_acc.png')
    plt.show()

    torch.save(G, 'conditional_gan_multi/Conditional_Generator_save.pth')
    print('Generator saved.')


"""
A function that loads a trained Generator model and uses it to create synthetic samples
"""


def generate_synthetic_samples(num_of_samples=100, num_of_features=310, label=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load initial data
    _, real_data, _ = prepare_data()

    generator = torch.load('conditional_gan_multi/Conditional_Generator_save.pth')
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
    scaler = joblib.load("conditional_gan/scaler.save")
    synthetic_data = scaler.inverse_transform(synthetic_samples)
    synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

    synthetic_samples = synthetic_data.copy(deep=True)
    # Insert column containing labels
    synthetic_data.insert(loc=310, column='label', value=class_labels, allow_duplicates=True)
    # Round values to closest integer for columns that should be boolean
    synthetic_data = transform_bool.transform(synthetic_data)
    # Map booleans to 1 and 0.
    synthetic_data = synthetic_data.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    # pickle.dump(synthetic_data, open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data' + str(num_of_samples), 'wb'))

    return synthetic_data, real_data


def generate_30000_samples_per_class():
    ## For each class, generate 30000 synthetic samples
    synthetic_data0, _ = generate_synthetic_samples(num_of_samples=30000, label=0)
    synthetic_data1, _ = generate_synthetic_samples(num_of_samples=30000, label=1)
    synthetic_data2, _ = generate_synthetic_samples(num_of_samples=30000, label=2)
    synthetic_data3, _ = generate_synthetic_samples(num_of_samples=30000, label=3)
    synthetic_data4, _ = generate_synthetic_samples(num_of_samples=30000, label=4)
    synthetic_data5, _ = generate_synthetic_samples(num_of_samples=30000, label=5)

    # List of above dataframes
    pdList = [synthetic_data0, synthetic_data1, synthetic_data2, synthetic_data3, synthetic_data4, synthetic_data5]
    final_df = pd.concat(pdList)

    # Shuffle the dataframe
    final_df = final_df.sample(frac=1)

    pickle.dump(final_df,
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_30000_per_class', 'wb'))
    return final_df


def generate_samples_to_reach_30000_per_class():
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
                open('../data/synthetic_data/conditional_gan_multiclass/synthetic_data_balanced_per_class', 'wb'))
    return final_df


def evaluate_synthetic_data(synthetic_data):
    real_data = pickle.load(open('../data/original_data/train_multiclass_data', 'rb'))

    print('\n~~~~~~~~~~~~~~ Synthetic Data Evaluation ~~~~~~~~~~~~~~')

    print('\n----------- Comparing synthetic to train real data ----------- ')
    ks = KSTest.compute(synthetic_data, real_data)
    print('Inverted Kolmogorov-Smirnov D statistic: {}'.format(ks))
    #kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
    #print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))

    real_data = pickle.load(open('../data/original_data/test_multiclass_data', 'rb'))
    print('\n----------- Comparing synthetic to test real data ----------- ')
    ks = KSTest.compute(synthetic_data, real_data)
    print('Inverted Kolmogorov-Smirnov D statistic: {}'.format(ks))
    #kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
    #print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))


#train_gan(epochs=300)

print('\n~~~~~~~~~~~~~~ Evaluating method of creating 30000 new samples for each class ~~~~~~~~~~~~~~')
evaluate_synthetic_data(synthetic_data=generate_30000_samples_per_class())
print('\n~~~~~~~~~~~~~~ Evaluating method of creating a balanced dataset ~~~~~~~~~~~~~~')
evaluate_synthetic_data(synthetic_data=generate_samples_to_reach_30000_per_class())
