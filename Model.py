import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LeakyReLU, Dropout, TimeDistributed, Embedding, concatenate, Bidirectional, LSTM
from keras import optimizers, losses, activations
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import backend as K


class NERModel(object):
    def __init__(self, word_length, labels, case_embeddings, pos_embedings, word_embeddings, char_embedding, char_case_embedding,
                 rnn_size=100, filters=25, pool_size=3, kernel_size=3, dropout=0.5, leaky_alpha=0.3):
        self.word_length = word_length  # 52

        self.leaky_alpha = leaky_alpha
        self.dropout = dropout
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.rnn_size = rnn_size

        self.labels = np.array(labels)
        self.case_embeddings = np.array(case_embeddings)
        self.pos_embedings = np.array(pos_embedings)
        self.word_embeddings = np.array(word_embeddings)
        self.char_embedding = np.array(char_embedding)
        self.char_case_embedding = np.array(char_case_embedding)

        self.num_labels = self.labels.shape[0]

    def build_model(self, loss='sparse_categorical_crossentropy', optimizer='nadam'):

        print("Building NER Model .... ")
        # Word Case Embeddings
        word_case_input = Input(shape=(None,))
        word_case_embedding = Embedding(
            input_dim=self.case_embeddings.shape[0],
            output_dim=self.case_embeddings.shape[1],
            weights=[self.case_embeddings],
            trainable=True)(word_case_input)

        print("Word Case Embedding : " + str(word_case_embedding.shape))

        # POS Embeddings
        pos_input = Input(shape=(None,))
        pos_embedding = Embedding(
            input_dim=self.pos_embedings.shape[0],
            output_dim=self.pos_embedings.shape[1],
            weights=[self.pos_embedings],
            trainable=True)(pos_input)

        print("POS Embedding : " + str(pos_embedding.shape))

        # Word Embeddings
        word_input = Input(shape=(None,))
        word_embedding = Embedding(
            input_dim=self.word_embeddings.shape[0],
            output_dim=self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            trainable=False)(word_input)

        print("Word Embedding : " + str(word_embedding.shape))

        # Char Embeddings
        char_input = Input(shape=(None, self.word_length))
        char_embedding = TimeDistributed(Embedding(
            input_dim=self.char_embedding.shape[0],
            output_dim=self.char_embedding.shape[1],
            weights=[self.char_embedding],
            trainable=True))(char_input)

        print("Char Embedding (Before CNN) : " + str(char_embedding.shape))

        char_cnn = TimeDistributed(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size))(char_embedding)
        char_cnn = TimeDistributed(LeakyReLU(alpha=self.leaky_alpha))(char_cnn)
        char_cnn = TimeDistributed(Dropout(rate=self.dropout))(char_cnn)
        char_cnn = TimeDistributed(MaxPooling1D(
            pool_size=self.pool_size))(char_cnn)
        print("Char Embedding (After CNN) : " + str(char_cnn.shape))
        char_cnn = TimeDistributed(Flatten())(char_cnn)
        print("Char Embedding (After Flatten) : " + str(char_cnn.shape))

        # Char Case Embeddings
        char_case_input = Input(shape=(None, self.word_length))
        char_case_embedding = TimeDistributed(Embedding(
            input_dim=self.char_case_embedding.shape[0],
            output_dim=self.char_case_embedding.shape[1],
            weights=[self.char_case_embedding],
            trainable=True))(char_case_input)

        print("Char Case Embedding (Before CNN) : " +
              str(char_case_embedding.shape))

        char_case_cnn = TimeDistributed(Conv1D(
            filters=self.filters, kernel_size=self.kernel_size))(char_case_embedding)
        char_case_cnn = TimeDistributed(
            LeakyReLU(alpha=self.leaky_alpha))(char_case_cnn)
        char_case_cnn = TimeDistributed(
            Dropout(rate=self.dropout))(char_case_cnn)
        char_case_cnn = TimeDistributed(MaxPooling1D(
            pool_size=self.pool_size))(char_case_cnn)
        print("Char Embedding (Before CNN) : " + str(char_case_cnn.shape))
        char_case_cnn = TimeDistributed(Flatten())(char_case_cnn)
        print("Char Case Embedding (After Flatten) : " + str(char_case_cnn.shape))

        rnn_input = concatenate(
            [word_case_embedding, pos_embedding, word_embedding, char_cnn, char_case_cnn])

        print("Final Input Tensor to LSTM : " + str(rnn_input.shape))

        # Bi-LSTM on Concatenated Output
        # 2 Bi-LSTM
        output = Bidirectional(
            LSTM(units=self.rnn_size, return_sequences=True))(rnn_input)
        output = Bidirectional(
            LSTM(units=self.rnn_size, return_sequences=True))(output)

        print("Output of Bi-LSTM : " + str(output.shape))

        output = TimeDistributed(Dense(self.num_labels))(output)
        print("Output of Dense Network : " + str(output.shape))

        model = Model(inputs=[word_case_input, pos_input, word_input,
                              char_input, char_case_input], outputs=[output])
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()
        plot_model(model, to_file='ner.png')
