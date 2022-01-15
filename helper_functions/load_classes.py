import pickle
import pandas as pd


def return_dataset_labels():
    labels = {'Pron': 'spambot', 'Traditional_Spambots_3': 'spambot', 'Traditional_Spambots_1': 'spambot',
              'Traditional_Spambots_4': 'spambot', 'twittertechnology': 'spambot',
              'Social_Spambots_1': 'socialbot', 'Social_Spambots_2': 'socialbot', 'Social_Spambots_3': 'socialbot',
              'fastfollowerz': 'socialbot', 'Cresci_Stock': 'socialbot',
              'Astroturf_political': 'politicalbot', 'Midterm': 'politicalbot', 'Political': 'politicalbot',
              'Vendor': 'socialbot',
              'News_Agency': 'cyborg', 'Company': 'cyborg', 'Celebrities': 'cyborg', 'botwiki': 'selfdeclaredbots',
              'Botometer': 'otherbots', 'Cresci_RTbust': 'otherbots', 'Gilani': 'otherbots',
              'intertwitter': 'otherbots', 'Varol': 'otherbots', 'Caverlee': 'spambot'}
    return labels


def return_user_specific_class():
    user_dataset = pickle.load(open('../data/original_data/user_dataset_mapping', 'rb'))
    userlabels = pickle.load(open('../data/original_data/user_labels_new', 'rb'))
    bot_datasets = return_dataset_labels()
    final_class = {}
    bot_class = {}
    bots = []
    humans = []
    for u, t in userlabels.items():
        if t == 'human':
            humans.append(u)
            final_class[u] = 'human'
        else:
            bots.append(u)
    j = 0
    for b in bots:
        d = user_dataset[b]
        try:
            clasi = bot_datasets[d]
            final_class[b] = clasi
            # print (b,clasi)
        except KeyError:
            print(d)
    from collections import Counter
    d = dict(Counter(final_class.values()))
    s = 0
    for k, v in d.items():
        s = s + int(v)
    return final_class


multi = return_user_specific_class()

data = pickle.load(open('../data/original_data/final_data_no_rts_v2', 'rb'))
df = pd.DataFrame(data)

df['label'] = df['user_id'].map(multi)
df = df[~df['label'].isin(['otherbots'])]

if 'max_appearance_of_punc_mark' in df.columns:
    df = df.drop(['max_appearance_of_punc_mark'], axis=1)
df['label'] = df['label'].map(
    {'human': 0, 'socialbot': 1, 'politicalbot': 2, 'spambot': 3, 'selfdeclaredbots': 4, 'cyborg': 5})

# summarize class distribution
print('Bot types distribution:' + '\n', df['label'].value_counts())

pickle.dump(df, open('../data/original_data/multi_class_data', 'wb'))

X = df.drop(['user_name', 'user_screen_name', 'user_id', 'label', 'time'], axis=1)
y = df['label']

