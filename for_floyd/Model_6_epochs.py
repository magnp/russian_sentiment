'''
Run in floyd hub

floyd run --gpu --env tensorflow-1.9 --data glycosylase2/datasets/ru_sent/1:data "python Model_6_epochs.py"

'''


# coding=utf-8
import string

import keras
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import pandas as pd
import numpy as np
#import data_util
import io
from keras.utils.np_utils import to_categorical
import math
from functools import reduce
from sklearn.metrics import precision_score, recall_score

import operator
import pickle

from sklearn.utils.class_weight import compute_class_weight

max_features = 20000
max_len = 20
batch_size = 256
epochs = 8
learning_rate = 5e-5
#lr_decay = 2e-2

cur_model_name = 'multiclass_bidir_2layer_16_emb_100_drop02everywhere3_3.h5'

''' Compute class weights '''

def get_data(data_path):
    df = pd.read_csv(data_path, delimiter=",")
    print(df.shape)
    #print(df['label'].unique())
    df = df[df['label'].isin(['-1', '1', '0', '2'])]
    print(df.shape)
    data = df.values[:, 1:]
    return data

data = get_data('/floyd/input/data/cleaned_data.csv')
y_true = data[:, 1]
class_weights = compute_class_weight('balanced', np.unique(y_true), y_true)
class_weights = {0: class_weights[0],
                 1: class_weights[1],
                 2: class_weights[2]}


''' From file data_util.py '''

def construct_good_set(data, top=300, skip=10):
    word_cnt = {}
    for sentence in list(data):
        for word in sentence:
            if word_cnt.__contains__(word):
                word_cnt[word] += 1
            else:
                word_cnt[word] = 1
    top_eng_words = sorted(word_cnt.items(), key=operator.itemgetter(1), reverse=True)
    return set([key for (key, value) in top_eng_words][skip:top + skip - 1])


def sentences_to_scalars_loaded_dict(data, good_set):
    words_to_nums = restore_words_to_nums()
    for i, sentence in enumerate(data):
        data[i] = list(filter(lambda x: x in good_set, sentence))
        for j, word in enumerate(data[i]):
            if not words_to_nums.__contains__(word):
                data[i][j] = 0
            else:
                data[i][j] = words_to_nums[word]


def sentences_to_scalars(data, good_set):
    words_to_nums = {}
    cur = 1
    for i, sentence in enumerate(data):
        data[i] = list(filter(lambda x: x in good_set, sentence))
        for j, word in enumerate(data[i]):
            if not words_to_nums.__contains__(word):
                words_to_nums[word] = cur
                cur += 1
            data[i][j] = words_to_nums[word]
            # data[i] = ' '.join(str(x) for x in data[i])
    save_words_to_nums(words_to_nums)


def save_words_to_nums(words_to_nums):
    afile = open(r'words_to_nums.pkl', 'wb')
    pickle.dump(words_to_nums, afile)
    afile.close()


def restore_words_to_nums():
    afile = open(r'words_to_nums.pkl', 'rb')
    words_to_nums = pickle.load(afile)
    afile.close()
    return words_to_nums


def sentences_to_predefined_scalars(data, good_set, map_to_scalar):
    for i, sentence in enumerate(data):
        data[i] = list(filter(lambda x: x in good_set, sentence))
        for j, word in enumerate(data[i]):
            if word in map_to_scalar:
                data[i][j] = map_to_scalar[word].index
            else:
                data[i][j] = 0


''' Train model '''                
                
                
def train_model():
    data = get_data('/floyd/input/data/cleaned_data.csv')
    np.random.seed(17)
    np.random.shuffle(data)

    xs, _ = get_xs(data)
    best_words_set = construct_good_set(xs, max_features, 0)
    sentences_to_scalars(xs, best_words_set)

    train_test_split = int(0.8 * len(xs))

    xs_train_scalar = keras.preprocessing.sequence.pad_sequences(xs[:train_test_split], maxlen=max_len, padding='post',
                                                                 truncating='post')
    xs_test_scalar = keras.preprocessing.sequence.pad_sequences(xs[train_test_split:], maxlen=max_len, padding='post',
                                                                truncating='post')

    ys_train, ys_test = get_ys(data, train_test_split)
    ys_train = to_categorical(ys_train)
    ys_test = to_categorical(ys_test)
    print(ys_train[:5,:])

    model = construct_model(max_features, max_len)

    model.fit(
        xs_train_scalar, ys_train,
        batch_size=batch_size,
        validation_data=(xs_test_scalar, ys_test),
        epochs=epochs,
        class_weight=class_weights
    )

    model.save(cur_model_name)


def get_xs(data):
    xs_str = data[:, 0]
    xs = np.copy(xs_str)
    xs = [str(x).encode().decode('utf-8').split(' ') for x in xs]
    return xs, xs_str


def get_ys(data, train_test_split):
    ys = data[:, 1]
    ys = np.where(ys == '-1', '0', ys)
    
    '''
    identity = str.maketrans('', '')
    allchars = ''.join(chr(i) for i in range(256))
    nondigits = allchars.translate(identity, string.digits + '-')
    # 
    #this code below is very ugly (it was build around edge cases), but it works.
    ys = [str(y).translate(identity, nondigits) for y in ys]
    ys = [y if len(y) > 0 else '2' for y in ys]
    ys = [0 if y == -1 else int(
        2 if y == '-' or (len(y) > 1 and y[1] == '-') or math.isnan(float(y))
            else (0 if y == '-1' else y)#.translate(identity, nondigits)
    ) for y in ys]
    '''
    
    ys_train = ys[:train_test_split]
    ys_test = ys[train_test_split:]
    return ys_train, ys_test


def construct_model(max_features, max_len):
    model = Sequential()

    model.add(Embedding(max_features, 100, input_length=max_len))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])
    print(model.summary())
    return model

train_model()