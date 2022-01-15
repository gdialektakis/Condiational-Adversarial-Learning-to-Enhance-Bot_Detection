import pandas as pd

"""
This function transforms specific column of a Dataframe from floats to integers by rounding 
to the closest integer and then to boolean.
"""


def transform(df):
    df['default_profile'] = df['default_profile'].map(lambda x: bool(round(x)))
    df['default_profile_image'] = df['default_profile_image'].map(lambda x: bool(round(x)))
    df['verified'] = df['verified'].map(lambda x: bool(round(x)))
    df['location'] = df['location'].map(lambda x: bool(round(x)))
    df['url'] = df['url'].map(lambda x: bool(round(x)))
    df['hashtags_in_name'] = df['hashtags_in_name'].map(lambda x: bool(round(x)))
    df['hashtags_in_description'] = df['hashtags_in_description'].map(lambda x: bool(round(x)))
    df['urls_in_description'] = df['urls_in_description'].map(lambda x: bool(round(x)))
    df['bot_word_in_name'] = df['bot_word_in_name'].map(lambda x: bool(round(x)))
    df['bot_word_in_screen_name'] = df['bot_word_in_screen_name'].map(lambda x: bool(round(x)))
    df['bot_word_in_description'] = df['bot_word_in_description'].map(lambda x: bool(round(x)))
    df['source_change'] = df['source_change'].map(lambda x: bool(round(x)))

    return df
