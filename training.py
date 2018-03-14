from Data import Data
from Model import NERModel

data = Data()
train_sentences = data.get_train_sentences()
val_sentence = data.get_val_sentences()
test_sentence = data.get_test_sentences()

train_sentences, word_length, _ = data.formated_dataset(train_sentences[:20])
val_sentence, _, _ = data.formated_dataset(val_sentence[:20], batch=True, batch_size=10)
test_sentence, _, _ = data.formated_dataset(test_sentence[:20], batch=False)

case_embeddings = data.get_case_embeddings()
pos_embedings = data.get_pos_embeddings()
word_embeddings = data.get_glove_embeddings()
char_embedding = data.get_char_embeddings()
char_case_embedding = data.get_char_case_embeddings()
labels = data.get_label_one_hot()

nerModel = NERModel(word_length=word_length, labels=labels,
                    case_embeddings=case_embeddings, pos_embedings=pos_embedings,
                    word_embeddings=word_embeddings,
                    char_embedding=char_embedding, char_case_embedding=char_case_embedding)
nerModel.build_model()
nerModel.train(inputs=train_sentences, validation_data=val_sentence,epochs=40)
