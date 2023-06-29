import requests
from bs4 import BeautifulSoup

# 딥러닝 모델을 통해 알아낸 의도, 개체명을 이용해서 DB에서 답변 검색


class FindAnswer:
    def __init__(self, db):
        self.db = db

    def make_query(self, intent_name, ner_tags):
        sql = "select * from chatbot_train_data"

        if intent_name != None and ner_tags == None:
            sql = sql + " where intent='{}'".format(intent_name)

        elif intent_name != None and ner_tags != None:
            where = " where intent='%s'" % intent_name
            if len(ner_tags) > 0:
                where += " and ("
                for ne in ner_tags:
                    where += " ner like '%{}%' or ".format(ne)
                where = where[:-3] + ")"
            sql = sql + where

        # 동일한 답변이 두개 이상인 경우, 랜덤으로 하나 선택
        sql = sql + " order by rand() limit 1"
        return sql

    def search(self, intent_name, ner_tags):
        # 의도, 개체명으로 검색
        sql = self.make_query(intent_name, ner_tags)
        answer = self.db.select_one(sql)

        # 만약 결과가 없으면
        # 의도명으로만 검색
        if answer is None:
            sql = self.make_query(intent_name, None)
            answer = self.db.select_one(sql)

        # 소괄호로 묶은건 튜플로 리턴
        return (answer["answer"], answer["answer_sub_url"])

    # ner태그를 실제 입력된 단어로 변환
    def tag_to_word(self, ner_predict, answer):
        for word, tag in ner_predict:
            # 변환해야 하는 태그가 있는 경우 추가
            if tag == "B_LC":
                answer = answer.replace(tag, word)

        answer = answer.replace("{", "")
        answer = answer.replace("}", "")
        return answer

    # airbnb 크롤링
    def search_hotel(self, entities):
        # entities = self.predict(query)  # 입력된 문장에 대해 NER 수행하여 entity 추출
        city_name = [entity[0] for entity in entities if entity[1] == "B_LC"]
        keywords = "+".join(city_name)

        url = f"https://www.airbnb.co.kr/s/{keywords}/homes?tab_id=home_tab"
        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36",
        # }

        response = requests.get(url)
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
                {"제목": title, "평점": score, "링크": "https://www.airbnb.co.kr" + link_href}
            )

        return results

    # 맛집 검색
    def search_rest(self, entities):
        # entities = self.predict(query)  # 입력된 문장에 대해 NER 수행하여 entity 추출
        city_name = [entity[0] for entity in entities if entity[1] == "B_LC"]
        keywords = "+".join(city_name)

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

            results.append({"제목": title, "평점": score, "링크": link})

        return results
