import operator
import pickle


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
