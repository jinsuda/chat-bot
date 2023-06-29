import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import numpy as np


class NerModel:
    def __init__(self, model_name, preprocess):
        # BIO 태그
        self.index_to_ner = {
            1: "O",
            2: "B_AIR",
            3: "I",
            4: "B_LC",
            5: "B_DT",
            6: "B_OG",
            7: "B_PS",
            8: "B_TI",
            9: "NNP",
            0: "PAD",
        }

        # 모델 불러오기
        self.model = load_model(model_name)

        # 전처리 객체
        self.p = preprocess

    def predict_tag(self, query):
        pos = self.p.pos(query)
        keyword = self.p.get_keyword(pos, without_tag=True)
        sequence = [self.p.get_wordindex_sequence(keyword)]
        pad_seq = preprocessing.sequence.pad_sequences(
            sequence, maxlen=40, padding="post"
        )
        # print("=======================================")
        # print(pad_seq)

        pred = self.model.predict(pad_seq)
        pred_class = np.argmax(pred, axis=-1)
        # print("=======================================")
        # print(pred_class)
        tag = []
        for tag_idx in pred_class[0]:
            # O태그 제외하고 나머지만 tag배열에 넣어줌
            if tag_idx == 1:
                continue
            tag.append(self.index_to_ner[tag_idx])

        if len(tag) == 0:
            return None
        return tag

    def predict(self, query):
        pos = self.p.pos(query)
        keyword = self.p.get_keyword(pos, without_tag=True)
        sequence = [self.p.get_wordindex_sequence(keyword)]

        pad_seq = preprocessing.sequence.pad_sequences(
            sequence, maxlen=40, padding="post"
        )

        pred = self.model.predict(pad_seq)
        pred_class = np.argmax(pred, axis=-1)

        tag = [self.index_to_ner[i] for i in pred_class[0]]
        return list(zip(keyword, tag))
