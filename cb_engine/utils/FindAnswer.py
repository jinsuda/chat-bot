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
