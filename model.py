# coding=utf-8
import string

import keras
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import data_util
import io
from keras.utils.np_utils import to_categorical
import math
from functools import reduce
from sklearn.metrics import precision_score, recall_score

max_features = 20000
max_len = 20
batch_size = 32
epochs = 1
cur_model_name = 'multiclass_bidir_2layer_16_emb_100_drop02everywhere3_1.h5'

def train_model():
    data = get_data('data/cleaned_data.csv')
    np.random.shuffle(data)

    xs, _ = get_xs(data)
    best_words_set = data_util.construct_good_set(xs, max_features, 0)
    data_util.sentences_to_scalars(xs, best_words_set)


    train_test_split = int(0.8 * len(xs))

    xs_train_scalar = keras.preprocessing.sequence.pad_sequences(xs[:train_test_split], maxlen=max_len, padding='post',
                                                                 truncating='post')
    xs_test_scalar = keras.preprocessing.sequence.pad_sequences(xs[train_test_split:], maxlen=max_len, padding='post',
                                                                truncating='post')

    ys_train, ys_test = get_ys(data, train_test_split)
    ys_train = to_categorical(ys_train)
    ys_test = to_categorical(ys_test)

    model = construct_model(max_features, max_len)

    model.fit(
        xs_train_scalar, ys_train,
        batch_size=batch_size,
        validation_data=(xs_test_scalar, ys_test),
        epochs=epochs
    )

    model.save(cur_model_name)


def get_xs(data):
    xs_str = data[:, 0]
    xs = np.copy(xs_str)
    xs = [str(x).encode().decode('utf-8').split(' ') for x in xs]
    return xs, xs_str


def test_model_on_labeled_data():
    data = get_data('data/cleaned_data.csv')

    xs, xs_str = get_xs(data)
    best_words_set = data_util.construct_good_set(xs, max_features, 0)
    data_util.sentences_to_scalars_loaded_dict(xs, best_words_set)

    xs_scalar = keras.preprocessing.sequence.pad_sequences(xs, maxlen=max_len, padding='post',
                                                                 truncating='post')

    ys = data[:, 1]
    #print(np.unique(ys.astype('str')))
    ys = [0 if y == '-1' else int(0 if math.isnan(float(y)) else y) for y in ys]
    model = load_model(cur_model_name)
    # print(xs_scalar.shape)
    # print(xs_scalar[:1,:], model.predict(xs_scalar[:1,:]))
    result = np.argmax(model.predict(xs_scalar), axis=1)
    np.savetxt('data/cleaned_predict_1.csv', result, delimiter=',')
    print(result)
    visual = zip(xs_str, result)
    predicted_and_true = zip(result, ys)
    true_pos = reduce(lambda a, b: a + (1 if b[0] == b[1] and b[0] == 1 else 0), predicted_and_true, 0)
    false_pos = reduce(lambda a, b: a + (1 if b[0] != b[1] and b[1] == 0 else 0), predicted_and_true, 0)
    false_neg = reduce(lambda a, b: a + (1 if b[0] != b[1] and b[1] == 1 else 0), predicted_and_true, 0)
    
    precision = true_pos / np.float(true_pos + false_neg)
    recall = true_pos / np.float(true_pos + false_pos)
    #precision = precision_score(ys, result)
    #recall = recall_score(ys, result)
    # print(precision)
    # print(recall)

def test_model_on_unlabeled_data():
    data = get_data('data/cleaned_data_ok.csv')

    xs, xs_str = get_xs(data)
    best_words_set = data_util.construct_good_set(xs, max_features, 0)

    data_util.sentences_to_scalars_loaded_dict(xs, best_words_set)

    xs_scalar = keras.preprocessing.sequence.pad_sequences(xs, maxlen=max_len, padding='post',
                                                                 truncating='post')

    model = load_model(cur_model_name)
    result = model.predict(xs_scalar)
    max_indices = [np.argmax(r) for r in result]
    # strings_and_scores = zip(xs_str, result)
    pos = []
    neg = []
    neutral = []

    for index, i in enumerate(max_indices):
        if i == 0:
            neg.append((xs_str[index], result[index]))
            continue
        if i == 1:
            pos.append((xs_str[index], result[index]))
            continue
        if i == 2:
            neutral.append((xs_str[index], result[index]))
            continue
        print("Something's wrong")
    pos = list(sorted(pos, key=lambda x: x[1][1]))
    neg = list(sorted(neg, key=lambda x: x[1][0]))
    neutral = list(sorted(neutral, key=lambda x: x[1][2]))

    # visual_sorted = list(sorted(strings_and_scores, key=lambda x:x[1]))
    def write_to_file(strings_and_scores, path):
        with io.open(path, 'w', encoding='utf-8') as file_handler:
            for item in strings_and_scores:
                st = str(item[0]).decode('utf-8')
                file_handler.write(u"{}\t{}\n".format(st, item[1]))

    write_to_file(pos, 'data/res_pos.txt')
    write_to_file(neg, 'data/res_neg.txt')
    write_to_file(neutral, 'data/res_neutral.txt')
    print(3)


def get_ys(data, train_test_split):
    ys = data[:, 1]
    identity = string.maketrans('', '')
    allchars = ''.join(chr(i) for i in xrange(256))
    nondigits = allchars.translate(identity, string.digits+'-')

    #this code below is very ugly (it was build around edge cases), but it works.
    ys = [str(y).translate(identity, nondigits) for y in ys]
    ys = [y if len(y) > 0 else '2' for y in ys]
    ys = [0 if y == -1 else int(
        2 if y == '-' or (len(y) > 1 and y[1] == '-') or math.isnan(float(y))
            else (0 if y == '-1' else y)#.translate(identity, nondigits)
    ) for y in ys]
    ys_train = ys[:train_test_split]
    ys_test = ys[train_test_split:]
    return ys_train, ys_test


def get_data(data_path):
    df = pd.read_csv(data_path, delimiter=",")
    print(df.shape)
    #print(df['label'].unique())
    df = df[df['label'].isin(['-1', '1', '0', '2'])]
    print(df.shape)
    data = df.values[:, 1:]
    return data


def construct_model(max_features, max_len):
    model = Sequential()

    model.add(Embedding(max_features, 100, input_length=max_len))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# train_model()
# test_model_on_labeled_data()
# test_model_on_unlabeled_data()