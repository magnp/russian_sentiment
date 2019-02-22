# coding=utf-8
import string

import re
import sklearn
import pandas as pd
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
#regex = re.compile('[%s\w]' % re.escape(string.punctuation))
i = 0
seen = set()

def text_cleaner(text, is_utf=True):
    if type(text) is str:
        text = text.lower()
        if not is_utf:
            text = text.encode().decode('utf-8')
        text = re.sub('[^А-Яа-я\-]', ' ', text)
        text = [morph.parse(word)[0].normal_form for word in text.split()]
        text = ' '.join(text)

    global i
    i += 1
    if i % 10000 == 0:
        print(i)
        print(text)
        
    return text


def prepare_twitter_dataset():
    positives = pd.read_csv('data/positive.csv', sep=';', header=None)
    negatives = pd.read_csv('data/negative.csv', sep=';', header=None)
    neutrals = pd.read_csv('data/pre_neutral.csv', sep=';', header=None, error_bad_lines=False, engine='python')

    dataset = pd.concat([positives, negatives, neutrals])
    # dataset = neutrals
    dataset = dataset[[3, 4]]
    dataset.columns = ['text', 'label']

    dataset['text'] = dataset['text'].apply(text_cleaner, is_utf=False)
    dataset.drop_duplicates('text', inplace=True)

    dataset.to_csv('data/cleaned_data.csv', encoding='utf-8')


def prepare_ok_dataset():
    dataset = pd.read_csv('data/train_content.csv', sep='\t', header=None, encoding='utf-8')
    dataset.columns = ['text']

    dataset['text'] = dataset['text'].apply(text_cleaner)

    dataset.to_csv('data/cleaned_data_ok.csv', encoding='utf-8')


# prepare_ok_dataset()
prepare_twitter_dataset()