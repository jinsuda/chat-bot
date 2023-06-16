import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np
import requests


class NerModel:
    def __init__(self, model_name, preprocess):
        # BIO 태그
        self.index_to_ner = {
            1: "O",
            2: "B_AIR",
            3: "I",
            4: "B_DT",
            5: "B_LC",
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

    # 호텔 검색
    def serch_hotel(self, query):
        entities = self.predict(query)  # 입력된 문장에 대해 NER 수행하여 entity 추출
        city_lc = [
            entity for entity in entities if entity[1] == "B_LC"
        ]  # 책 관련 entity 필터링
        city_name = [entity[0] for entity in city_lc]  # 추출된 책 이름들
        # print(city_name)
        url = "https://hotels4.p.rapidapi.com/locations/v3/search"
        querystring = {
            "q": city_name,
            "locale": "ko_KR",
        }

        headers = {
            "X-RapidAPI-Key": "477cad2fb7msh45fdc7b31b5ac16p1396b1jsn79ba71a45fc4",
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com",
        }

        response = requests.get(url, headers=headers, params=querystring)

        response_json = response.json()
        sr_list = response_json["sr"]  # sr 리스트 가져오기

        display_names = []  # displayName 값을 저장할 리스트

        for sr_item in sr_list:
            if sr_item["type"] == "HOTEL":
                display_name = sr_item["regionNames"]["shortName"]
                display_names.append(display_name)

        return display_names
