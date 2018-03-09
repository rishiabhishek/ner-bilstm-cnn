import pandas as pd
import numpy as np
from keras.utils import to_categorical
import os
import string


class Data(object):

    def __init__(self):
        print("Read Dataset")
        # self.data_path = os.path.join(os.path.expanduser(
        #     '~'), "OtherWorkspace/Important-Datasets/conll2003/eng.train")
        # self.embedding_path = os.path.join(os.path.expanduser(
        #     '~'), "OtherWorkspace/Embedding/glove.6B/glove.6B.100d.txt")

        self.data_path = os.path.join(os.path.expanduser(
            '~'), "Workspace/Important-Datasets/conll2003/eng.train")
        self.embedding_path = os.path.join(os.path.expanduser(
            '~'), "Workspace/Datasets/embeddings/glove.6B/glove.6B.100d.txt")

        self.dataset, self.words, self.pos, self.labels = self.read_file()

        self.features = {"allCaps": 0, "upperInitial": 1, "lowercase": 2,
                         "mixedCaps": 3, "number": 4, "noinfo": 5}
        self.one_hot = np.identity(len(self.features))

        print("Create POS Features...")
        self.pos_features, self.pos_onehot = self.create_pos_features()
        print("No. of POS : " + str(len(self.pos_features)))

        print("Create Labels...")
        # self.labels = set(self.dataset[:, -1])
        self.label_features = {ne: i for i, ne in enumerate(self.labels)}
        self.label_onehot = np.identity(len(self.labels))

        print("Create Character Encoding...")
        self.char_vocab, self.char_onehot = self.char_level_feature()

        print("Additional Char-Level Features")
        self.additional_feature = {"uppercase": 0,
                                   "lowercase": 1, "punctuation": 2, "other": 3, "PADDING": 4}

        self.add_feature_one_hot = np.identity(len(self.additional_feature))

        print("Loading GloVe Embeddings....")
        embedding_lines = open(self.embedding_path,
                               mode='r', encoding="utf-8").readlines()
        glove_embeddings = {}
        for line in embedding_lines:
            items = line.replace("\n", "").split(" ")
            glove_embeddings[items[0]] = np.array(items[1:]).astype(np.float)

        print("Create Word Vocabulary...")
        self.vocab, self.embeddings = self.create_vocab(glove_embeddings)

        print("Vocab Size : " + str(len(self.vocab)))
        # Parameters
        self.word_length = [len(word) for word in self.words]
        self.char_num = max(self.word_length)
        print("Max number of Characters in a Word: " + str(self.char_num))

        self.sent_length = [len(sent) for sent in self.dataset]
        self.word_num = max(self.sent_length)
        print("Max number of Words in a Sentence: " + str(self.word_num))
        print("Character Feature length : " +
              str(len(self.char_vocab) + len(self.additional_feature)))

    def get_sentences(self):
        return self.dataset

    def get_case_embeddings(self):
        return self.one_hot

    def get_pos_embeddings(self):
        return self.pos_onehot

    def get_char_embeddings(self):
        return self.char_onehot

    def get_char_case_embeddings(self):
        return self.add_feature_one_hot

    def get_glove_embeddings(self):
        return self.embeddings

    def get_label_one_hot(self):
        return self.label_onehot

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

    def create_vocab(self, glove_embeddings):
        dimension = len(list(glove_embeddings.values())[0])
        words = sorted(set([str(word).lower() for word in self.words]))
        vocab = {word: i for i, word in enumerate(words)}
        embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), dimension))
        for word, i in vocab.items():
            if word in glove_embeddings:
                embeddings[i, :] = glove_embeddings[word]
            # else:
            #     embeddings[i, :]=np.random.uniform(-0.25, 0.25, dimension)

        return vocab, embeddings

    def isUpperInitial(self, word):
        upperInitial = word[0].isupper()
        if upperInitial:
            for i in range(1, len(word)):
                if not word[i].islower():
                    return False
            return True

    def isMixedCaps(self, word):
        return any(letter.islower() for letter in word) and any(letter.isupper() for letter in word)

    def get_caps_feature(self, word):
        if word.isupper():
            return self.features["allCaps"]
        elif word.islower():
            return self.features["lowercase"]
        elif word.isdigit():
            return self.features["number"]
        elif self.isUpperInitial(word):
            return self.features["upperInitial"]
        elif self.isMixedCaps(word):
            return self.features["mixedCaps"]
        else:
            return self.features["noinfo"]

    def create_pos_features(self):
        # pos_set = set(self.dataset[:, 1])
        pos_features = {pos: i for i, pos in enumerate(self.pos)}
        pos_onehot = np.identity(len(pos_features))
        return pos_features, pos_onehot

    def get_pos_features(self, pos):
        return self.pos_features[pos]

    def get_labels(self, label):
        return self.label_features[label]

    def char_level_feature(self):
        char_set = set()
        for word in self.words:
            for char in str(word):
                char_set.add(char)

        char_set = sorted(char_set)
        char_vocab = {char: i for i, char in enumerate(char_set)}
        char_vocab["PADDING"] = len(char_vocab)
        char_vocab["UNKNOWN"] = len(char_vocab)
        return char_vocab, np.identity(len(char_vocab))

    def get_char_level_feature(self, char):
        if char in self.char_vocab:
            return self.char_vocab[char]
        else:
            return self.char_vocab["UNKNOWN"]

    def get_additional_char_feature(self, char):
        if char.isupper():
            return self.additional_feature["uppercase"]
        elif char.islower():
            return self.additional_feature["lowercase"]
        elif char in string.punctuation:
            return self.additional_feature["punctuation"]
        else:
            return self.additional_feature["other"]

    def get_embbeding(self, word):
        return self.vocab[word]

    def encode_sentences(self, sentences):
        encoded_sentences = []
        for sentence in sentences:
            case_feature = []
            pos_feature = []
            char_feature = []
            char_case_feature = []
            word_embeding = []
            labels = []
            for word, pos, label in sentence:
                # Word Case
                case_feature.append(self.get_caps_feature(word))
                # POS of Word
                pos_feature.append(self.get_pos_features(pos))

                # Word Embedding
                word_embeding.append(self.get_embbeding(word.lower()))
                # Char level features
                char_level_feature = []
                add_char_level_feature = []
                for char in word:
                    char_level_feature.append(
                        self.get_char_level_feature(char))
                    add_char_level_feature.append(
                        self.get_additional_char_feature(char))

                char_feature.append(self.padding(char_level_feature))
                char_case_feature.append(self.padding(add_char_level_feature))

                labels.append(self.get_labels(label))
            encoded_sentences.append([
                case_feature, pos_feature, char_feature, char_case_feature, word_embeding, labels])

        return encoded_sentences

    # Convert Word to char level features
    def padding(self, features):
        feature_len = len(features)
        padding_len = self.char_num - feature_len
        paddings = [self.additional_feature["PADDING"]
                    for _ in range(padding_len)]
        return features + paddings


# data = Data()
# sentences = data.get_sentences()
# encoded_sentences = data.encode_sentences(sentences)
# print(encoded_sentences[0])
