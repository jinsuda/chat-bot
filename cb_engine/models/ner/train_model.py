import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from utils.preprocessing import Preprocessing


def read_file(file_name):
    sent = []
    with open(file_name, "r", encoding="utf8") as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ";" and lines[idx + 1][0] == "$":
                this_sent = []
            elif l[0] == "$" and lines[idx - 1][0] == ";":
                continue
            elif l[0] == "\n":
                sent.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
        return sent


p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)

corpus = read_file("cb_engine/models/ner/ner_train.txt")
# print(corpus)
sentence = []
tag = []
for t in corpus:
    sen = []
    bio = []

    for w in t:
        # print(w)
        sen.append(w[1])
        bio.append(w[3])
    sentence.append(sen)
    tag.append(bio)


tag_tokenizer = keras.preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tag)

voca_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print(tag_tokenizer.word_index)
# 단어 시퀀스
x_train = [p.get_wordindex_sequence(sent) for sent in sentence]
y_train = tag_tokenizer.texts_to_sequences(tag)

index_to_ner = tag_tokenizer.index_word
index_to_ner[0] = "PAD"

x_train = keras.utils.pad_sequences(x_train, padding="post", maxlen=40)
y_train = keras.utils.pad_sequences(y_train, padding="post", maxlen=40)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# 원핫인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)


model = keras.Sequential()

model.add(
    keras.layers.Embedding(
        input_dim=voca_size, output_dim=30, input_length=40, mask_zero=True
    )
)

model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25
        )
    )
)

model.add(
    keras.layers.TimeDistributed(keras.layers.Dense(tag_size, activation="softmax"))
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
model.fit(x_train, y_train, batch_size=128, epochs=10)

model.save("ner_model.h5")

# f1 score -> 모델 겅증하고 싶을 때
