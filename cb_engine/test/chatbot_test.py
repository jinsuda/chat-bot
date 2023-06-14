from configs.DatabaseConfig import *
from utils.Database import Database
from utils.preprocessing import Preprocessing

p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)

# 데이터베이스 삭제
db = Database(
    host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME, port=DB_PORT
)
db.connect()

query = "오늘 대한항공 예약 가능할까요?"

# 의도 파악
from models.intent.intentModel import IntentModel

intent = IntentModel(model_name="cb_engine/models/intent/intent_model.h5", preprocess=p)
predict = intent.predict_class(query)
intent_name = intent.label[predict]

# 개체명 인식
from models.ner.NerModel import NerModel

ner = NerModel(model_name="cb_engine/models/ner/ner_model.h5", preprocess=p)
ner_predict = ner.predict(query)
tag = ner.predict_tag(query)

print("질문 : ", query)
print("==================================")
print("의도 : ", intent_name)
print("개체명 : ", ner_predict)
print("ner태그 :", tag)


# ner_predict로  if문 써서 org품사에 '항공'인지 '호텔'인지 확인
# model_ner_test.py 에서 predict랑 태그 확인

# 답변 검색
from utils.FindAnswer import FindAnswer

try:
    f = FindAnswer(db)
    answer_text, answer_image = f.search(intent_name, tag)
    answer = f.tag_to_word(ner_predict, answer_text)
except:
    answer = "무슨 말인지 모르겠어요"

print("==================================")
print("답변 : ", answer)
db.close()
