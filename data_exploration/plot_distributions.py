import helper_scripts.feature_taxonomy as ft
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

pd.options.mode.chained_assignment = None


def dimensionality_reduction():
    data = pickle.load(open('../data/multi_class_data', 'rb'))
    data = data.reset_index(drop=True)
    # summarize class distribution
    print('Bot types distribution:' + '\n', data['label'].value_counts())

    user_features, temporal_features, content_features, sentiment_features, \
    hashtag_network_features = ft.group_features_no_rts(df=data)

    # Normalize our data before applying PCA
    user_features = StandardScaler().fit_transform(X=user_features)
    temporal_features = StandardScaler().fit_transform(X=temporal_features)
    content_features = StandardScaler().fit_transform(X=content_features)
    sentiment_features = StandardScaler().fit_transform(X=sentiment_features)
    hashtag_network_features = StandardScaler().fit_transform(X=hashtag_network_features)

    user_features_reduced = PCA(n_components=1).fit_transform(user_features)
    temporal_features_reduced = PCA(n_components=1).fit_transform(temporal_features)
    content_features_reduced = PCA(n_components=1).fit_transform(content_features)
    sentiment_features_reduced = PCA(n_components=1).fit_transform(sentiment_features)
    hashtag_network_features_reduced = PCA(n_components=1).fit_transform(hashtag_network_features)

    # Concatenate above feature arrays into a single Dataframe
    features_df = pd.DataFrame(np.concatenate(
        [user_features_reduced, temporal_features_reduced, content_features_reduced, sentiment_features_reduced,
         hashtag_network_features_reduced], axis=1), columns=['user_features', 'temporal_features', 'content_features',
                                                              'sentiment_features', 'hashtag_network_features'])

    features_df['label'] = data['label']
    return features_df


def divide_df(df):
    """
    Divide the initial dataframe to sub-dataframes each containing a specific bot class.
    Bot classes:
    0 -> Human
    1 -> Social Bot
    2 -> Political Bot
    3 -> Spam Bot
    4 -> Self-declared
    5 -> Cyborg
    """
    human = df[df['label'] == 0]
    social_bot = df[df['label'] == 1]
    political_bot = df[df['label'] == 2]
    spam_bot = df[df['label'] == 3]
    self_declared_bot = df[df['label'] == 4]
    cyborg = df[df['label'] == 5]

    human = human.drop(['label'], axis=1)
    social_bot = social_bot.drop(['label'], axis=1)
    political_bot = political_bot.drop(['label'], axis=1)
    spam_bot = spam_bot.drop(['label'], axis=1)
    self_declared_bot = self_declared_bot.drop(['label'], axis=1)
    cyborg = cyborg.drop(['label'], axis=1)
    return human, social_bot, political_bot, spam_bot, self_declared_bot, cyborg


def plot_CDF(df):
    human, social_bot, political_bot, spam_bot, self_declared_bot, cyborg = divide_df(df)
    # User features
    human_user_features = human[['user_features']]
    social_bot_user_features = social_bot[['user_features']]
    political_bot_user_features = political_bot[['user_features']]
    spam_bot_user_features = spam_bot[['user_features']]
    self_declared_bot_user_features = self_declared_bot[['user_features']]
    cyborg_bot_user_features = cyborg[['user_features']]

    # Content features
    human_content_features = human[['content_features']]
    social_bot_content_features = social_bot[['content_features']]
    political_bot_content_features = political_bot[['content_features']]
    spam_bot_content_features = spam_bot[['content_features']]
    self_declared_bot_content_features = self_declared_bot[['content_features']]
    cyborg_bot_content_features = cyborg[['content_features']]

    # Temporal features
    human_temporal_features = human[['temporal_features']]
    social_bot_temporal_features = social_bot[['temporal_features']]
    political_bot_temporal_features = political_bot[['temporal_features']]
    spam_bot_temporal_features = spam_bot[['temporal_features']]
    self_declared_bot_temporal_features = self_declared_bot[['temporal_features']]
    cyborg_bot_temporal_features = cyborg[['temporal_features']]

    # Sentiment features
    human_sentiment_features = human[['sentiment_features']]
    social_bot_sentiment_features = social_bot[['sentiment_features']]
    political_bot_sentiment_features = political_bot[['sentiment_features']]
    spam_bot_sentiment_features = spam_bot[['sentiment_features']]
    self_declared_bot_sentiment_features = self_declared_bot[['sentiment_features']]
    cyborg_bot_sentiment_features = cyborg[['sentiment_features']]

    # Hashtag network features
    human_hashtag_network_features = human[['hashtag_network_features']]
    social_bot_hashtag_network_features = social_bot[['hashtag_network_features']]
    political_bot_hashtag_network_features = political_bot[['hashtag_network_features']]
    spam_bot_hashtag_network_features = spam_bot[['hashtag_network_features']]
    self_declared_bot_hashtag_network_features = self_declared_bot[['hashtag_network_features']]
    cyborg_bot_hashtag_network_features = cyborg[['hashtag_network_features']]

    ############################# User features #############################
    # Plot the CDF for user features for all types of bots
    plt.figure()
    human_user_features['cdf'] = human_user_features.rank(method='average', pct=True)
    ax1 = human_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True,
                                                                title='User features CDF', label='human', xlim=(-5, 20),
                                                                linestyle='dashdot')

    social_bot_user_features['cdf'] = social_bot_user_features.rank(method='average', pct=True)
    social_bot_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True,
                                                               ax=ax1, label='social_bot', linestyle='dashdot')

    political_bot_user_features['cdf'] = political_bot_user_features.rank(method='average', pct=True)
    political_bot_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True,
                                                                  ax=ax1, label='political_bot', linestyle='dashdot')
    spam_bot_user_features['cdf'] = spam_bot_user_features.rank(method='average', pct=True)
    spam_bot_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True, ax=ax1,
                                                             label='spam_bot', linestyle='dashdot')
    self_declared_bot_user_features['cdf'] = self_declared_bot_user_features.rank(method='average', pct=True)
    self_declared_bot_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True, ax=ax1,
                                                                      label='self_declared', linestyle='dashdot')

    cyborg_bot_user_features['cdf'] = cyborg_bot_user_features.rank(method='average', pct=True)
    cyborg_bot_user_features.sort_values('user_features').plot(x='user_features', y='cdf', grid=True, ax=ax1,
                                                               label='cyborg', linestyle='dashdot')
    # plt.show()
    plt.savefig("cdf_plots/user_features_CDF.jpg")

    ############################# Content features #############################
    # Plot the CDF for user features for all types of bots
    plt.figure()
    human_content_features['cdf'] = human_content_features.rank(method='average', pct=True)
    ax1 = human_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True,
                                                                title='Content features CDF', label='human', linestyle='dashdot')

    social_bot_content_features['cdf'] = social_bot_content_features.rank(method='average', pct=True)
    social_bot_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True,
                                                               ax=ax1, label='social_bot', linestyle='dashdot')

    political_bot_content_features['cdf'] = political_bot_content_features.rank(method='average', pct=True)
    political_bot_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True,
                                                                  ax=ax1, label='political_bot', linestyle='dashdot')
    spam_bot_content_features['cdf'] = spam_bot_content_features.rank(method='average', pct=True)
    spam_bot_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True, ax=ax1,
                                                             label='spam_bot', linestyle='dashdot')
    self_declared_bot_content_features['cdf'] = self_declared_bot_content_features.rank(method='average', pct=True)
    self_declared_bot_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True, ax=ax1,
                                                                      label='self_declared', linestyle='dashdot')

    cyborg_bot_content_features['cdf'] = cyborg_bot_content_features.rank(method='average', pct=True)
    cyborg_bot_content_features.sort_values('content_features').plot(x='content_features', y='cdf', grid=True, ax=ax1,
                                                               label='cyborg', linestyle='dashdot')
    #plt.show()
    plt.savefig("cdf_plots/content_features_CDF.jpg")

    ############################# Temporal features #############################
    # Plot the CDF for user features for all types of bots
    plt.figure()
    human_temporal_features['cdf'] = human_temporal_features.rank(method='average', pct=True)
    ax1 = human_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True,
                                                                      title='Temporal features CDF', label='human', xlim=(-10, 50), linestyle='dashdot')

    social_bot_temporal_features['cdf'] = social_bot_temporal_features.rank(method='average', pct=True)
    social_bot_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True,
                                                                     ax=ax1, label='social_bot', linestyle='dashdot')

    political_bot_temporal_features['cdf'] = political_bot_temporal_features.rank(method='average', pct=True)
    political_bot_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True,
                                                                        ax=ax1, label='political_bot', linestyle='dashdot')
    spam_bot_temporal_features['cdf'] = spam_bot_temporal_features.rank(method='average', pct=True)
    spam_bot_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True, ax=ax1,
                                                                   label='spam_bot', linestyle='dashdot')
    self_declared_bot_temporal_features['cdf'] = self_declared_bot_temporal_features.rank(method='average', pct=True)
    self_declared_bot_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True,
                                                                            ax=ax1,
                                                                            label='self_declared', linestyle='dashdot')

    cyborg_bot_temporal_features['cdf'] = cyborg_bot_temporal_features.rank(method='average', pct=True)
    cyborg_bot_temporal_features.sort_values('temporal_features').plot(x='temporal_features', y='cdf', grid=True, ax=ax1,
                                                                     label='cyborg', linestyle='dashdot')
    #plt.show()
    plt.savefig("cdf_plots/temporal_features_CDF.jpg")

    ############################# Sentiment features #############################
    # Plot the CDF for user features for all types of bots
    plt.figure()
    human_sentiment_features['cdf'] = human_sentiment_features.rank(method='average', pct=True)
    ax1 = human_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True,
                                                                        title='Sentiment features CDF', label='human', xlim=(-10, 60), linestyle='dashdot')

    social_bot_sentiment_features['cdf'] = social_bot_sentiment_features.rank(method='average', pct=True)
    social_bot_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True,
                                                                       ax=ax1, label='social_bot', linestyle='dashdot')

    political_bot_sentiment_features['cdf'] = political_bot_sentiment_features.rank(method='average', pct=True)
    political_bot_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True,
                                                                          ax=ax1, label='political_bot', linestyle='dashdot')
    spam_bot_sentiment_features['cdf'] = spam_bot_sentiment_features.rank(method='average', pct=True)
    spam_bot_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True, ax=ax1,
                                                                     label='spam_bot', linestyle='dashdot')
    self_declared_bot_sentiment_features['cdf'] = self_declared_bot_sentiment_features.rank(method='average', pct=True)
    self_declared_bot_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True,
                                                                              ax=ax1,
                                                                              label='self_declared', linestyle='dashdot')

    cyborg_bot_sentiment_features['cdf'] = cyborg_bot_sentiment_features.rank(method='average', pct=True)
    cyborg_bot_sentiment_features.sort_values('sentiment_features').plot(x='sentiment_features', y='cdf', grid=True,
                                                                       ax=ax1,
                                                                       label='cyborg', linestyle='dashdot')

    #plt.show()
    plt.savefig("cdf_plots/sentiment_features_CDF.jpg")

    ############################# Hashtag Network features #############################
    # Plot the CDF for user features for all types of bots
    plt.figure()
    human_hashtag_network_features['cdf'] = human_hashtag_network_features.rank(method='average', pct=True)
    ax1 = human_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf', grid=True,
                                                                          title='Hashtag Network features CDF', label='human', xlim=(-5, 10), linestyle='dashdot')

    social_bot_hashtag_network_features['cdf'] = social_bot_hashtag_network_features.rank(method='average', pct=True)
    social_bot_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf', grid=True,
                                                                         ax=ax1, label='social_bot', linestyle='dashdot')

    political_bot_hashtag_network_features['cdf'] = political_bot_hashtag_network_features.rank(method='average', pct=True)
    political_bot_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf', grid=True,
                                                                            ax=ax1, label='political_bot', linestyle='dashdot')
    spam_bot_hashtag_network_features['cdf'] = spam_bot_hashtag_network_features.rank(method='average', pct=True)
    spam_bot_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf', grid=True,
                                                                       ax=ax1,
                                                                       label='spam_bot', linestyle='dashdot')
    self_declared_bot_hashtag_network_features['cdf'] = self_declared_bot_hashtag_network_features.rank(method='average', pct=True)
    self_declared_bot_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf',
                                                                                grid=True,
                                                                                ax=ax1,
                                                                                label='self_declared', linestyle='dashdot')

    cyborg_bot_hashtag_network_features['cdf'] = cyborg_bot_hashtag_network_features.rank(method='average', pct=True)
    cyborg_bot_hashtag_network_features.sort_values('hashtag_network_features').plot(x='hashtag_network_features', y='cdf', grid=True,
                                                                         ax=ax1,
                                                                         label='cyborg', linestyle='dashdot')
    #plt.show()
    plt.savefig("cdf_plots/hashtag_network_features_CDF.jpg")
    return


def plot_PDF(df):
    human, social_bot, political_bot, spam_bot, self_declared_bot, cyborg = divide_df(df)

    ############################# User features #############################
    plt.figure()
    ax1 = human.user_features.plot.density(color='blue', xlim=(-20, 20), linewidth=0.8, linestyle='dashdot')
    social_bot.user_features.plot.density(color='orange', ax=ax1, linewidth=0.8, linestyle='dashdot')
    political_bot.user_features.plot.density(color='green', ax=ax1, linewidth=0.8, linestyle='dashdot')
    spam_bot.user_features.plot.density(color='red', ax=ax1, linewidth=0.8, linestyle='dashdot')
    self_declared_bot.user_features.plot.density(color='purple', ax=ax1, linewidth=0.8, linestyle='dashdot')
    cyborg.user_features.plot.density(color='brown', ax=ax1, linewidth=0.8, linestyle='dashdot')
    plt.title('Probability Density plot for User features')
    ax1.legend(['human', 'social_bot', 'political_bot', 'spam_bot', 'self_declared_bot', 'cyborg'])
    #plt.show()
    plt.savefig("pdf_plots/user_features_PDF.jpg")

    ############################# Content features #############################
    plt.figure()
    ax1 = human.content_features.plot.density(color='blue', xlim=(-30, 30), linewidth=0.8, linestyle='dashdot')
    social_bot.content_features.plot.density(color='orange', ax=ax1, linewidth=0.8, linestyle='dashdot')
    political_bot.content_features.plot.density(color='green', ax=ax1, linewidth=0.8, linestyle='dashdot')
    spam_bot.content_features.plot.density(color='red', ax=ax1, linewidth=0.8, linestyle='dashdot')
    self_declared_bot.content_features.plot.density(color='purple', ax=ax1, linewidth=0.8, linestyle='dashdot')
    cyborg.content_features.plot.density(color='brown', ax=ax1, linewidth=0.8, linestyle='dashdot')
    plt.title('Probability Density plot for Content features')
    ax1.legend(['human', 'social_bot', 'political_bot', 'spam_bot', 'self_declared_bot', 'cyborg'])
    #plt.show()
    plt.savefig("pdf_plots/content_features_PDF.jpg")

    ############################# Temporal features #############################
    plt.figure()
    ax1 = human.temporal_features.plot.density(color='blue', xlim=(-40, 40), linewidth=0.8, linestyle='dashdot')
    social_bot.temporal_features.plot.density(color='orange', ax=ax1, linewidth=0.8, linestyle='dashdot')
    political_bot.temporal_features.plot.density(color='green', ax=ax1, linewidth=0.8, linestyle='dashdot')
    spam_bot.temporal_features.plot.density(color='red', ax=ax1, linewidth=0.8, linestyle='dashdot')
    self_declared_bot.temporal_features.plot.density(color='purple', ax=ax1, linewidth=0.8, linestyle='dashdot')
    cyborg.temporal_features.plot.density(color='brown', ax=ax1, linewidth=0.8, linestyle='dashdot')
    plt.title('Probability Density plot for Temporal features')
    ax1.legend(['human', 'social_bot', 'political_bot', 'spam_bot', 'self_declared_bot', 'cyborg'])
    #plt.show()
    plt.savefig("pdf_plots/temporal_features_PDF.jpg")

    ############################# Sentiment features #############################
    plt.figure()
    ax1 = human.sentiment_features.plot.density(color='blue', xlim=(-20, 20), linewidth=0.8, linestyle='dashdot')
    social_bot.sentiment_features.plot.density(color='orange', ax=ax1, linewidth=0.8, linestyle='dashdot')
    political_bot.sentiment_features.plot.density(color='green', ax=ax1, linewidth=0.8, linestyle='dashdot')
    spam_bot.sentiment_features.plot.density(color='red', ax=ax1, linewidth=0.8, linestyle='dashdot')
    self_declared_bot.sentiment_features.plot.density(color='purple', ax=ax1, linewidth=0.8, linestyle='dashdot')
    cyborg.sentiment_features.plot.density(color='brown', ax=ax1, linewidth=0.8, linestyle='dashdot')
    plt.title('Probability Density plot for Sentiment features')
    ax1.legend(['human', 'social_bot', 'political_bot', 'spam_bot', 'self_declared_bot', 'cyborg'])
    #plt.show()
    plt.savefig("pdf_plots/sentiment_features_PDF.jpg")

    ############################# Hashtag Network features #############################
    plt.figure()
    ax1 = human.hashtag_network_features.plot.density(color='blue', xlim=(-10, 10), linewidth=0.8, linestyle='dashdot')
    social_bot.hashtag_network_features.plot.density(color='orange', ax=ax1, linewidth=0.8, linestyle='dashdot')
    political_bot.hashtag_network_features.plot.density(color='green', ax=ax1, linewidth=0.8, linestyle='dashdot')
    spam_bot.hashtag_network_features.plot.density(color='red', ax=ax1, linewidth=0.8, linestyle='dashdot')
    self_declared_bot.hashtag_network_features.plot.density(color='purple', ax=ax1, linewidth=0.8, linestyle='dashdot')
    cyborg.hashtag_network_features.plot.density(color='brown', ax=ax1, linewidth=0.8, linestyle='dashdot')
    plt.title('Probability Density plot for Hashtag Network features')
    ax1.legend(['human', 'social_bot', 'political_bot', 'spam_bot', 'self_declared_bot', 'cyborg'])
    #plt.show()
    plt.savefig("pdf_plots/hashtag_network_features_PDF.jpg")
    return


def plot_ccdf(df):
    return


features_df = dimensionality_reduction()

plot_CDF(features_df)

plot_PDF(features_df)