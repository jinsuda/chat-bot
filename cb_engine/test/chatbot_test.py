from configs.DatabaseConfig import *
from utils.Database import Database
from utils.preprocessing import Preprocessing
from utils.FindAnswer import FindAnswer
from models.ner.NerModel import NerModel
from models.intent.intentModel import IntentModel

p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)
while True:
    # 데이터베이스 삭제
    db = Database(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME, port=DB_PORT
    )
    db.connect()

    query = input("질문: ")

    # 의도 파악

    intent = IntentModel(
        model_name="cb_engine/models/intent/intent_model.h5", preprocess=p
    )
    predict = intent.predict_class(query)
    intent_name = intent.label[predict]

    # 개체명 인식
    ner = NerModel(model_name="cb_engine/models/ner/ner_model.h5", preprocess=p)
    ner_predict = ner.predict(query)
    tag = ner.predict_tag(query)

    print("==================================")
    print("의도 : ", intent_name)
    print("개체명 : ", ner_predict)
    print("ner태그 :", tag)

    # 답변 검색
    try:
        f = FindAnswer(db)
        answer_text, answer_image = f.search(intent_name, tag)
        answer = f.tag_to_word(ner_predict, answer_text)
        if intent_name == "정보":
            result = ner.serch_hotel(query)
    except:
        answer = "무슨 말인지 모르겠어요"

    print("==================================")
    print("답변 : ", answer, result)
    db.close()
