import helper_scripts.feature_taxonomy as ft
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


def dimensionality_reduction(data=pickle.load(open('../data/multi_class_data', 'rb'))):
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