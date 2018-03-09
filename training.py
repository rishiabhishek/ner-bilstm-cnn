from Data import Data
from Model import NERModel


data = Data()
sentences = data.get_sentences()
encoded_sentences = data.encode_sentences(sentences)


case_embeddings = data.get_case_embeddings()
pos_embedings = data.get_pos_embeddings()
word_embeddings = data.get_glove_embeddings()
char_embedding = data.get_char_embeddings()
char_case_embedding = data.get_char_case_embeddings()
labels = data.get_label_one_hot()


nerModel = NERModel(word_length=52, labels=labels, case_embeddings=case_embeddings, pos_embedings=pos_embedings, word_embeddings=word_embeddings,
                    char_embedding=char_embedding, char_case_embedding=char_case_embedding)
nerModel.build_model()
