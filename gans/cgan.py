"""
Import necessary libraries to create a generative adversarial network
The code is developed using the PyTorch library
"""
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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
import helper_functions.transform_booleans as transform_bool
import warnings
warnings.filterwarnings('ignore')

"""
Network Architectures
The following are the discriminator and generator architectures
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_classes = 2
        self.num_features = 309
        self.prob = 0.2
        # embedding layer of the class labels (num_of_classes * encoding_size of each word)
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.num_features+self.num_classes, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.prob),
            nn.Linear(400, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.prob),
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
        self.num_classes = 2
        self.num_features = 309
        self.noise = 128
        # embedding layer of the class labels (num_of_classes * encoding_size of each word)
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.noise+self.num_classes, 500),
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


def prepare_data(batch_size=512, bots=True):
    bots_df = pickle.load(open('../binary_data/train_old_data_bots', 'rb'))
    humans_df = pickle.load(open('../binary_data/train_old_data_humans', 'rb'))

    # Concatenate human and bot training samples
    pdList = [bots_df, humans_df]
    df = pd.concat(pdList)

    # Shuffle the dataframe
    df = df.sample(frac=1)

    #df = df.sample(n=1000)
    y = df['label']

    if bots:
        df_filtered = df[df['label'] == 1]
    else:
        df_filtered = df[df['label'] == 0]

    df = df.drop(['label'], axis=1)
    df_filtered = df_filtered.drop(['label'], axis=1)

    # Scale our data in the range of (0, 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(X=df)

    # Store scaler for later use
    scaler_filename = "conditional_gan/scaler.save"
    joblib.dump(scaler, scaler_filename)

    # Transform dataframe into pytorch Tensor
    train = TensorDataset(torch.Tensor(df_scaled), torch.Tensor(np.array(y)))
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    return train_loader, df, df_filtered


def train_gan(n_epochs=100):
    """
    Determine if any GPUs are available
    """
    cuda = True if torch.cuda.is_available() else False

    """
    Hyperparameter settings
    """
    G_lr = 0.0002
    D_lr = 0.0002
    bs = 256

    # loss = nn.BCELoss()
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    num_of_classes = 2

    # Model
    G = Generator()
    D = Discriminator()

    if cuda:
        G.cuda()
        D.cuda()
        adversarial_loss.cuda()

    optimizer_G = optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.999))

    # Load our data
    train_loader, _, _ = prepare_data(batch_size=bs)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    """
    Network training procedure
    Every step both the loss for disciminator and generator is updated
    Discriminator aims to classify reals and fakes
    Generator aims to generate bot accounts as realistic as possible
    """
    mean_D_loss = []
    mean_G_loss = []
    D_acc = []
    for epoch in range(n_epochs):
        acc = []
        epoch_D_loss = []
        epoch_G_loss = []
        # labels = train_loader[1]
        for idx, train_data in enumerate(train_loader):
            idx += 1

            # Adversarial ground truths
            valid = Variable(FloatTensor(bs, 1).fill_(1.0), requires_grad=False).cuda()
            fake = Variable(FloatTensor(bs, 1).fill_(0.0), requires_grad=False).cuda()

            # Configure input
            real_inputs = Variable(train_data[0].type(FloatTensor))
            labels = Variable(train_data[1].type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            latent_dim = 128
            z = Variable(FloatTensor(np.random.normal(0, 1, (bs, latent_dim)))).cuda()
            gen_labels = Variable(LongTensor(np.random.randint(0, num_of_classes, bs)))

            # Generate a batch of bot samples
            gen_bots = G(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = D(gen_bots, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real bots
            validity_real = D(real_inputs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = D(gen_bots.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            """
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, idx, len(train_loader), d_loss.item(), g_loss.item())
            )
            """
            if idx % 100 == 0 or idx == len(train_loader):
                epoch_D_loss.append(d_loss.item())
                epoch_G_loss.append(g_loss.item())
                # D_accuracy = accuracy_score(labels, predictions)
                # acc.append(D_accuracy)
        if epoch > 291:
            torch.save(G, 'conditional_gan/CGAN_Generator' + str(epoch) + '.pth')
        # print('Epoch {} -- Discriminator mean Accuracy: {:.5f}'.format(epoch, mean(acc)))
        print('Epoch {} -- Discriminator mean loss: {:.5f}'.format(epoch, mean(epoch_D_loss)))
        print('Epoch {} -- Generator mean loss: {:.5f}'.format(epoch, mean(epoch_G_loss)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        mean_D_loss.append(mean(epoch_D_loss))
        mean_G_loss.append(mean(epoch_G_loss))
        # D_acc.append(mean(acc))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(mean_D_loss, color='blue', label='Discriminator loss')
    plt.plot(mean_G_loss, color='red', label='Generator loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('conditional_gan/cond_gan_loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(D_acc, color='blue', label='Discriminator accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('conditional_gan/cond_discriminator_acc.png')
    plt.show()

    torch.save(G, 'conditional_gan/Conditional_Generator_save.pth')
    print('Generator saved.')


"""
A function that loads a trained Generator model and uses it to create synthetic samples
"""


def generate_synthetic_samples(num_of_samples=100, num_of_features=309, num_of_classes=2, bots=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load initial data
    _, _, real_data = prepare_data(bots=bots)

    generator = torch.load('conditional_gan/CGAN_Generator296.pth')
    # Generate points in the latent space
    noise = (torch.rand(num_of_samples, 128) - 0.5) / 0.5
    noise = noise.to(device)

    # Create class labels
    if bots:
        class_labels = torch.randint(1, num_of_classes, (num_of_samples,)).to(device)
    else:
        class_labels = torch.randint(0, num_of_classes-1, (num_of_samples,)).to(device)

    # Pass latent points and class labels through our Generator to produce synthetic samples
    synthetic_samples = generator(noise, class_labels)

    # Transform pytorch tensor to numpy array
    synthetic_samples = synthetic_samples.cpu().detach().numpy()
    synthetic_samples = synthetic_samples.reshape(num_of_samples, num_of_features)
    class_labels = class_labels.cpu().detach().numpy()
    class_labels = class_labels.reshape(num_of_samples, 1)

    # Load saved min_max_scaler for inverse transformation of the generated data
    scaler = joblib.load("conditional_gan/scaler.save")
    synthetic_data = scaler.inverse_transform(synthetic_samples)
    synthetic_data = pd.DataFrame(data=synthetic_data, columns=real_data.columns)

    synthetic_samples = synthetic_data.copy(deep=True)
    # Insert column containing labels
    synthetic_data.insert(loc=309, column='label', value=class_labels, allow_duplicates=True)

    synthetic_data = transform_bool.transform(synthetic_data)

    return synthetic_data


def create_final_synthetic_dataset(test=False):
    if test:
        synthetic_data_bots = generate_synthetic_samples(num_of_samples=int(13688/2), bots=True)
        synthetic_data_humans = generate_synthetic_samples(num_of_samples=int(13688/2), bots=False)
    else:
        synthetic_data_bots = generate_synthetic_samples(num_of_samples=10548*2, bots=True)
        synthetic_data_humans = generate_synthetic_samples(num_of_samples=6553*2, bots=False)

    # Concatenate human and bot synthetic samples
    pdList = [synthetic_data_bots, synthetic_data_humans]
    final_df = pd.concat(pdList)

    # Shuffle the dataframe
    final_df = final_df.sample(frac=1)
    if test:
        pickle.dump(final_df, open('../binary_data/synthetic_data/cgan/synthetic_binary_test_data', 'wb'))
    else:
        pickle.dump(final_df, open('../binary_data/synthetic_data/cgan/synthetic_binary_data', 'wb'))
    return final_df


def evaluate_synthetic_data():
    real_data = pickle.load(open('../binary_data/train_old_data', 'rb'))
    real_data = real_data.drop(['label'], axis=1)

    print('\n~~~~~~~~~~~~~~ Synthetic Data Evaluation ~~~~~~~~~~~~~~')

    synthetic_data = pickle.load(open('../binary_data/synthetic_data/cgan/synthetic_binary_data', 'rb'))

    synthetic_data = synthetic_data.drop(['label'], axis=1)

    ks = KSTest.compute(synthetic_data, real_data)
    print('Inverted Kolmogorov-Smirnov D statistic on Train Data: {}'.format(ks))

    kl_divergence = evaluate(synthetic_data, real_data, metrics=['ContinuousKLDivergence'])
    print('Continuous Kullback–Leibler Divergence: {}'.format(kl_divergence))


#train_gan(n_epochs=300)
create_final_synthetic_dataset()
#evaluate_synthetic_data()
