import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LeakyReLU, Dropout
from keras import optimizers, losses, activations
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K


class NERModel(object):
    def __init__(self):
        self.batch_size = 100
        self.sentence_length = 110
        self.word_feature_length = 180
        self.word_length = 52
        self.char_feature_length = 90

        self.leaky_alpha = 0.3
        self.dropout = 0.5

    def build_model(self):

        cnn_input = Input(shape=(self.word_length, self.sentence_length))
        cnn = Conv1D(filters=25, kernel_size=7)(cnn_input)
        cnn = LeakyReLU(alpha=self.leaky_alpha)(cnn)
        cnn = Dropout(rate=self.dropout)(cnn)


