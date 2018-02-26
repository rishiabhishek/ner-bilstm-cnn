import pandas as pd
import numpy as np
from keras.utils import to_categorical


class Data(object):

    def __init__(self):
        self.data_path = "~/Workspace/Datasets/ner/ner_dataset.csv"
        self.embedding = ""
        self.dataset = pd.read_csv(
            self.data_path, encoding="iso-8859-1", error_bad_lines=False)
        self.words = self.dataset["Word"].values
        self.vocab = self.create_vocab()
        self.features = {"allCaps": 1, "upperInitial": 2, "lowercase": 3,
                         "mixedCaps": 4, "noinfo": 5}
        self.one_hot = np.identity(5)
        self.pos_features, self.pos_onehot = self.create_pos_features()

    def create_vocab(self):
        return set([word.lower() for word in self.words])

    def create_pos_features(self):
        pos_set = set(self.dataset["POS"].values.tolist())
        pos_features = {pos: i for i, pos in enumerate(pos_set)}
        pos_onehot = to_categorical(list(pos_features.values()))
        return pos_features, pos_onehot

    def get_pos_features(self, pos):
        self.pos_onehot[self.pos_features[pos] - 1]

    def get_caps_feature(self, word):
        if word.isupper():
            return self.one_hot[self.features["allCaps"] - 1]
        elif word.islower():
            return self.one_hot[self.features["lowercase"] - 1]
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


data = Data()
# features, one_hot = data.create_caps_feature()

print(data.create_pos_features()[0])
