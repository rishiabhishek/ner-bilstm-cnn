import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LeakyReLU, Dropout, TimeDistributed, Embedding, \
    concatenate, Bidirectional, LSTM
from keras import optimizers, losses, activations
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.utils import generic_utils
from keras import backend as K
import keras


class NERModel(object):
    def __init__(self, word_length, sentence_length, labels, case_embeddings, pos_embedings, word_embeddings,
                 char_embedding,
                 char_case_embedding,
                 rnn_size=200, filters=25, pool_size=3, kernel_size=3, dropout=0.5, leaky_alpha=0.3):

        self.word_length = word_length  # 52
        self.sentence_length = sentence_length  # 110

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
        self.model = None

        # GPU Config
        # cfg = K.tf.ConfigProto()
        # cfg.gpu_options.allow_growth = True
        # K.set_session(K.tf.Session(config=cfg))

    def build_model(self, loss='categorical_crossentropy', optimizer='nadam'):

        print("Building NER Model .... ")
        # Word Case Embeddings
        word_case_input = Input(shape=(self.sentence_length,), name="word_case")
        word_case_embedding = Embedding(
            input_dim=self.case_embeddings.shape[0],
            output_dim=self.case_embeddings.shape[1],
            weights=[self.case_embeddings],
            trainable=True)(word_case_input)

        print("Word Case Embedding : " + str(word_case_embedding.shape))

        # POS Embeddings
        pos_input = Input(shape=(self.sentence_length,), name="pos")
        pos_embedding = Embedding(
            input_dim=self.pos_embedings.shape[0],
            output_dim=self.pos_embedings.shape[1],
            weights=[self.pos_embedings],
            trainable=True)(pos_input)

        print("POS Embedding : " + str(pos_embedding.shape))

        # Word Embeddings
        word_input = Input(shape=(self.sentence_length,), name="word_embedding")
        word_embedding = Embedding(
            input_dim=self.word_embeddings.shape[0],
            output_dim=self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            trainable=False)(word_input)

        print("Word Embedding : " + str(word_embedding.shape))

        # Char Embeddings
        char_input = Input(shape=(self.sentence_length, self.word_length), name="char_embedding")
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
        char_case_input = Input(shape=(self.sentence_length, self.word_length), name="char_case")
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

        self.model = Model(inputs=[word_case_input, pos_input, word_input,
                                   char_input, char_case_input], outputs=[output])
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='ner.png')

    def _minibatch(self, dataset, batch_size):
        batch_num = int(len(dataset) / batch_size)
        for i in range(batch_num):
            word_case_inputs = []
            pos_inputs = []
            word_inputs = []
            char_inputs = []
            char_case_inputs = []
            labels_ = []
            start = i * batch_size
            end = i + batch_size
            for j in range(start, end):
                case_feature, pos_feature, char_feature, char_case_feature, word_embeding, labels = dataset[j]
                word_case_inputs.append(case_feature)
                pos_inputs.append(pos_feature)
                word_inputs.append(word_embeding)
                char_inputs.append(char_feature)
                char_case_inputs.append(char_case_feature)
                labels_.append(labels)

            word_case_inputs = np.array(word_case_inputs)
            pos_inputs = np.array(pos_inputs)
            word_inputs = np.array(word_inputs)
            char_inputs = np.array(char_inputs)
            char_case_inputs = np.array(char_case_inputs)
            labels_ = np.array(labels_)

            yield word_case_inputs, pos_inputs, word_inputs, char_inputs, char_case_inputs, labels_

    def train(self, inputs, train_on_batch=False, epochs=20, batch_size=50, validation_split=0.3):

        if train_on_batch:

            for epoch in range(epochs):
                print("Epoch %d/%d" % (epoch, epochs))
                a = generic_utils.Progbar(batch_size)
                # print("No. of Batches : " + str(len(batches)))
                for i, batch in enumerate(self._minibatch(inputs, batch_size)):
                    word_case_input, pos_input, word_input, char_input, char_case_input, labels_ = batch
                    print("Training batch : " + str(i))
                    self.model.train_on_batch(
                        [word_case_input, pos_input, word_input, char_input, char_case_input],
                        labels_)
                    # print(self.model.metrics_names)
                    a.update(i)
                    print(' ')
                print("Saving Model.....")
                self.model.save('ner_model.h5')
        else:
            word_case_inputs = []
            pos_inputs = []
            word_inputs = []
            char_inputs = []
            char_case_inputs = []
            labels_ = []

            for data in inputs:
                case_feature, pos_feature, char_feature, char_case_feature, word_embeding, labels = data
                word_case_inputs.append(case_feature)
                pos_inputs.append(pos_feature)
                word_inputs.append(word_embeding)
                char_inputs.append(char_feature)
                char_case_inputs.append(char_case_feature)
                labels_.append(labels)

            word_case_inputs = np.array(word_case_inputs)
            pos_inputs = np.array(pos_inputs)
            word_inputs = np.array(word_inputs)
            char_inputs = np.array(char_inputs)
            char_case_inputs = np.array(char_case_inputs)
            labels_ = np.array(labels_)

            self.model.fit(x=[word_case_inputs, pos_inputs, word_inputs, char_inputs, char_case_inputs], y=labels_,
                           epochs=epochs, batch_size=batch_size, validation_split=validation_split)
            print("Saving Model.....")
            self.model.save('ner_model.h5')
