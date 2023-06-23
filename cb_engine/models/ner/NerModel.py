import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np
import requests
from bs4 import BeautifulSoup
from redis_data import create_redis_client
import json

redis_client = create_redis_client()


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

    # airbnb 크롤링
    def search_hotel(self, query):
        entities = self.predict(query)  # 입력된 문장에 대해 NER 수행하여 entity 추출
        city_lc = [entity for entity in entities if entity[1] == "B_LC"]
        city_name = [entity[0] for entity in city_lc]  # 추출된 도시 이름들
        keywords = "+".join(city_name)

        # Redis에서 캐시된 데이터 확인
        cache_key = f"keywords_hotel:{keywords}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            # 캐시된 데이터가 존재하는 경우, 해당 데이터 반환
            decoded_data = cached_data.decode("utf-8")
            decoded_data = json.loads(decoded_data)
            return decoded_data

        url = f"https://www.airbnb.co.kr/s/{keywords}/homes?tab_id=home_tab"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        tags = soup.find_all(attrs={"data-testid": "listing-card-title"})
        subs = soup.find_all("span", class_="t1a9j9y7 r4a59j5 dir dir-ltr")
        links = soup.find_all("a", class_="l1ovpqvx bn2bl2p dir dir-ltr")

        results = []
        for tag, sub, link in zip(tags, subs, links):
            title = tag.get_text(strip=True)
            score = sub.get_text(strip=True)
            link_href = link["href"]

            results.append(
                {"숙소": title, "평점": score, "링크": "https://www.airbnb.co.kr" + link_href}
            )
        # 결과를 Redis에 캐시 저장
        redis_client.set(cache_key, json.dumps(results, ensure_ascii=False))

        return results

    # 맛집 검색
    def search_rest(self, query):
        entities = self.predict(query)  # 입력된 문장에 대해 NER 수행하여 entity 추출
        city_lc = [
            entity for entity in entities if entity[1] == "B_LC"
        ]  # 도시 관련 entity 필터링
        city_name = [entity[0] for entity in city_lc]  # 추출된 도시 이름들
        keywords = "+".join(city_name)

        # Redis에서 캐시된 데이터 확인
        cache_key = f"keywords_rest:{keywords}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            # 캐시된 데이터가 존재하는 경우, 해당 데이터 반환
            decoded_data = cached_data.decode("utf-8")
            decoded_data = json.loads(decoded_data)
            return decoded_data

        url = f"https://www.siksinhot.com/search?keywords={keywords}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
        }

        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.content, "html.parser")

        results = []
        for element in soup.find_all("a", class_="textBox"):
            title_element = element.find("h2")
            score_element = element.find(class_="score")

            title = title_element.get_text(strip=True) if title_element else None
            score = score_element.get_text(strip=True) if score_element else None

            link = element["href"] if "href" in element.attrs else None

            results.append({"식당": title, "평점": score, "링크": link})

        redis_client.set(cache_key, json.dumps(results, ensure_ascii=False))
        return results

    # 환율 정보
    # def search_exchange(self, query):
    #     entities = self.predict(query)
    #     country_lc = [entity for entity in entities if entity[1] == "B_LC"]
    #     country_name = [entity[0] for entity in country_lc]
    #     country = "+".join(country_name)

    #     url = "https://finance.naver.com/marketindex/?tabSel=exchange#tab_section"

    #     response = requests.get(url)
    #     soup = BeautifulSoup(response.text, "html.parser")

    #     option_elements = soup.find_all("option")
    #     exchange_rates = {}

    #     for option_element in option_elements:
    #         country_unit = option_element.get_text(strip=True)
    #         value = option_element["value"]
    #         country, unit = country_unit.split(" ", 1)
    #         exchange_rates[country] = {"value": value, "unit": unit}

    #     if country in exchange_rates:
    #         data = exchange_rates[country]
    #         value = data["value"]
    #         unit = data["unit"]
    #         return f"{country}: 1{unit} = {value}원 "
    #     else:
    #         return "해당 나라의 환율 정보를 찾을 수 없습니다."
