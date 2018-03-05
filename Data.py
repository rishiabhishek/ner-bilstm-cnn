import pandas as pd
import numpy as np
from keras.utils import to_categorical
import os


class Data(object):

    def __init__(self):
        print("Read Dataset")
        self.data_path = os.path.join(os.path.expanduser(
            '~'), "Workspace/Important-Datasets/conll2003/eng.train")
        self.embedding_path = os.path.join(os.path.expanduser(
            '~'), "Workspace/Datasets/embeddings/glove.6B/glove.6B.100d.txt")

        self.dataset, self.words, self.pos, self.labels = self.read_file()

        # print(self.dataset[0:20])

        print("Create Word Vocabulary...")
        self.vocab = self.create_vocab()
        self.features = {"allCaps": 1, "upperInitial": 2, "lowercase": 3,
                         "mixedCaps": 4, "number": 5, "noinfo": 6}
        self.one_hot = np.identity(len(self.features))

        print("Create POS Features...")
        self.pos_features, self.pos_onehot = self.create_pos_features()

        print("Create Labels...")
        # self.labels = set(self.dataset[:, -1])
        self.label_features = {ne: i for i, ne in enumerate(self.labels)}
        self.label_onehot = np.identity(len(self.labels))

        print("Create Character Encoding...")
        self.char_vocab, self.char_onehot = self.char_level_feature()

        print("Loading GloVe Embeddings....")
        embedding_lines = open(self.embedding_path,
                               mode='r', encoding="utf-8").readlines()
        self.embeddings = {}
        for line in embedding_lines:
            items = line.replace("\n", "").split(" ")
            self.embeddings[items[0]] = np.array(items[1:]).astype(np.float)

        # Parameters
        self.word_length = [len(word) for word in self.words]
        self.char_num = max(self.word_length)

        self.sent_length = [len(sent) for sent in self.dataset]
        self.word_num = max(self.sent_length)

    def create_vocab(self):
        return set([str(word).lower() for word in self.words])

    def create_pos_features(self):
        # pos_set = set(self.dataset[:, 1])
        pos_features = {pos: i for i, pos in enumerate(self.pos)}
        pos_onehot = np.identity(len(pos_features))
        return pos_features, pos_onehot

    def get_pos_features(self, pos):
        return self.pos_onehot[self.pos_features[pos] - 1]

    def get_labels(self, label):
        return self.label_onehot[self.label_features[label] - 1]

    def get_caps_feature(self, word):
        if word.isupper():
            return self.one_hot[self.features["allCaps"] - 1]
        elif word.islower():
            return self.one_hot[self.features["lowercase"] - 1]
        elif word.isdigit():
            return self.one_hot[self.features["number"] - 1]
        elif self.isUpperInitial(word):
            return self.one_hot[self.features["upperInitial"] - 1]
        elif self.isMixedCaps(word):
            return self.one_hot[self.features["mixedCaps"] - 1]
        else:
            return self.one_hot[self.features["noinfo"] - 1]

    def isUpperInitial(self, word):
        upperInitial = word[0].isupper()
        if upperInitial:
            for i in range(1, len(word)):
                if not word[i].islower():
                    return False
            return True

    def isMixedCaps(self, word):
        return any(letter.islower() for letter in word) and any(letter.isupper() for letter in word)

    def char_level_feature(self):
        char_set = set()
        for word in self.words:
            for char in str(word):
                char_set.add(char)

        char_set = sorted(char_set)
        # char_list = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
        char_vocab = {char: i for i, char in enumerate(char_set)}
        char_vocab["PADDING"] = len(char_vocab)
        char_vocab["UNKNOWN"] = len(char_vocab)
        return char_vocab, np.identity(len(char_vocab))

    def get_char_level_feature(self, char):
        if char in self.char_vocab:
            return self.char_onehot[self.char_vocab[char] - 1]
        else:
            return self.char_onehot[self.char_vocab["UNKNOWN"] - 1]

    def get_embbeding(self, word):
        if word.lower() in self.embeddings:
            return self.embeddings[word.lower()]
        else:
            return np.random.randn(100)

    def get_feature_word(self, word):
        char_feature_input = np.zeros((self.char_num, len(self.char_vocab)))
        for i in range(len(word)):
            char = word[i]
            char_feature_input[i, :] = self.get_char_level_feature(char)

        if self.char_num - len(word) > 0:
            padding = np.zeros(
                (self.char_num - len(word), len(self.char_vocab)))
            for i in range(self.char_num - len(word)):
                padding[i, :] = self.char_onehot[self.char_vocab["PADDING"] - 1]
            char_feature_input[len(word):, :] = padding

        return char_feature_input

    def read_file(self):
        lines = open(self.data_path, encoding="utf-8")
        sentences = []
        sentence = []
        pos = set()
        labels = set()
        words = set()
        for line in lines:
            if len(line) == 0 or line[0] == "-" or line[0] == "\n":
                if len(sentence) > 0:
                    sentences.append(np.array(sentence))
                    sentence = []
            else:
                items = line.split(' ')
                sentence.append(
                    [items[0], items[1], items[-1].replace("\n", "")])
                words.add(items[0])
                pos.add(items[1])
                labels.add(items[-1].replace("\n", ""))
        return sentences, words, pos, labels


data = Data()

print(data.get_feature_word("cat"))
# print("\n\n")
# print(data.get_embbeding("ahlk"))
