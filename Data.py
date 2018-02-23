import pandas as pd
import numpy as np


class Data(object):

    def __init__(self):
        self.data_path = "/Users/abhishekpradhan/OtherWorkspace/Datasets/ner_dataset.csv"
        self.embedding = ""
        self.dataset = pd.read_csv(self.data_path, encoding="iso-8859-1")
        self.words = self.dataset["Word"].values
        self.vocab = self.create_vocab()

    def create_vocab(self):
        return set([word.lower() for word in self.words])

    def create_caps_feature(self):
        features = {"allCaps": 1, "upperInitial": 2, "lowercase": 3,
                    "mixedCaps": 4, "noinfo": 5}

        one_hot = np.identity(5)
        return features, one_hot


data = Data()
features, one_hot = data.create_caps_feature()

print(one_hot[features["upperInitial"]-1])