import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LeakyReLU, Dropout, TimeDistributed, Embedding, \
    concatenate, Bidirectional, LSTM
from keras.models import Model
from keras.utils import generic_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score


class NERModel(object):
    def __init__(self, word_length, labels, case_embeddings, pos_embedings, word_embeddings,
                 char_embedding,
                 char_case_embedding,
                 rnn_size=275, filters=53, pool_size=3, kernel_size=3, dropout=0.68, leaky_alpha=0.3,
                 learning_rate=0.0105):

        self.word_length = word_length  # 52

        self.leaky_alpha = leaky_alpha
        self.dropout = dropout
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.rnn_size = rnn_size
        self.learning_rate = learning_rate

        self.labels = np.array(labels)
        self.case_embeddings = np.array(case_embeddings)
        self.pos_embedings = np.array(pos_embedings)
        self.word_embeddings = np.array(word_embeddings)
        self.char_embedding = np.array(char_embedding)
        self.char_case_embedding = np.array(char_case_embedding)

        self.num_labels = self.labels.shape[0]
        self.model = None

        # GPU Config
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))

    def build_model(self, loss='categorical_crossentropy', optimizer='sgd'):

        print("Building NER Model .... ")
        # Word Case Embeddings
        word_case_input = Input(shape=(None,), name="word_case")
        word_case_embedding = Embedding(
            input_dim=self.case_embeddings.shape[0],
            output_dim=self.case_embeddings.shape[1],
            weights=[self.case_embeddings],
            trainable=True)(word_case_input)

        print("Word Case Embedding : " + str(word_case_embedding.shape))

        # POS Embeddings
        pos_input = Input(shape=(None,), name="pos")
        pos_embedding = Embedding(
            input_dim=self.pos_embedings.shape[0],
            output_dim=self.pos_embedings.shape[1],
            weights=[self.pos_embedings],
            trainable=True)(pos_input)

        print("POS Embedding : " + str(pos_embedding.shape))

        # Word Embeddings
        word_input = Input(shape=(None,), name="word_embedding")
        word_embedding = Embedding(
            input_dim=self.word_embeddings.shape[0],
            output_dim=self.word_embeddings.shape[1],
            weights=[self.word_embeddings],
            trainable=False)(word_input)

        print("Word Embedding : " + str(word_embedding.shape))

        # Char Embeddings
        char_input = Input(shape=(None, self.word_length), name="char_embedding")
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
        char_case_input = Input(shape=(None, self.word_length), name="char_case")
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
        # output = Bidirectional(
        #     LSTM(units=self.rnn_size, return_sequences=True))(output)

        print("Output of Bi-LSTM : " + str(output.shape))

        output = TimeDistributed(Dense(self.num_labels, activation='softmax'))(output)
        print("Output of Dense Network : " + str(output.shape))

        self.model = Model(inputs=[word_case_input, pos_input, word_input,
                                   char_input, char_case_input], outputs=[output])

        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='ner.png')

    def _minibatch(self, dataset):
        batches = []
        for batch in dataset:
            word_case_inputs = []
            pos_inputs = []
            word_inputs = []
            char_inputs = []
            char_case_inputs = []
            labels_ = []
            for case_feature, pos_feature, char_feature, char_case_feature, word_embeding, labels in batch:
                word_case_inputs.append(case_feature)
                pos_inputs.append(pos_feature)
                word_inputs.append(word_embeding)
                char_inputs.append(char_feature)
                char_case_inputs.append(char_case_feature)
                labels_.append(labels)

            batches.append((np.array(word_case_inputs), np.array(pos_inputs), np.array(word_inputs), np.array(
                char_inputs), np.array(char_case_inputs), np.array(labels_)))
            # batches.append(final_batch)
        return batches

    def _validation_generator(self, val_data):
        word_case_inputs = []
        pos_inputs = []
        word_inputs = []
        char_inputs = []
        char_case_inputs = []
        labels_ = []
        for batch in val_data:
            for case_feature, pos_feature, char_feature, char_case_feature, word_embeding, labels in batch:
                word_case_inputs.append(case_feature)
                pos_inputs.append(pos_feature)
                word_inputs.append(word_embeding)
                char_inputs.append(char_feature)
                char_case_inputs.append(char_case_feature)
                labels_.append(labels)

        return np.array(word_case_inputs), np.array(pos_inputs), np.array(word_inputs), np.array(
            char_inputs), np.array(char_case_inputs), np.array(labels_)

    def convet_to_prediction(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        precision, recall, fscore, support = score(y_true, y_pred)
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1-Score: {}'.format(fscore))
        print('support: {}'.format(support))

    def train(self, inputs, validation_data, epochs=40):

        validation_data = self._validation_generator(validation_data)
        batches = self._minibatch(inputs)
        print("Batch Length : " + str(len(batches)))

        for epoch in range(epochs):

            print("Epoch {} / {}".format(epoch + 1, epochs))
            progbar = generic_utils.Progbar(len(batches))
            for i, batch in enumerate(batches):
                word_case_inputs, pos_inputs, word_inputs, char_inputs, char_case_inputs, labels_ = batch

                loss, acc = self.model.train_on_batch(
                    x=[word_case_inputs, pos_inputs, word_inputs, char_inputs, char_case_inputs],
                    y=labels_)

                progbar.add(1, values=[("train loss", loss), ("acc", acc)])

            val_word_case_inputs, val_pos_inputs, val_word_inputs, val_char_inputs, val_char_case_inputs, y_true = validation_data
            y_pred = self.model.predict(
                [val_word_case_inputs, val_pos_inputs, val_word_inputs, val_char_inputs, val_char_case_inputs],
                batch_size=10)

            self.convet_to_prediction(y_pred=y_pred, y_true=y_true)
            if epoch != 0 and epoch % 10 == 0:
                file_path = 'ner-model-{}.h5'.format(epoch + 1)
                self.model.save(file_path)
