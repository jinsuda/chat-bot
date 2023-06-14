from utils.preprocessing import Preprocessing
from models.ner.NerModel import NerModel
from konlpy.tag import Komoran

p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)

ner = NerModel(model_name="cb_engine/models/ner/ner_model.h5", preprocess=p)
query = "오늘 대한항공 한국 예매 하고싶어"
komoran = Komoran()
print(komoran.pos(query))
predict = ner.predict(query)
tag = ner.predict_tag(query)

print(predict)
print(tag)


# 오늘 날씨 알려줘
# 의도 -> 정보
# ner
# ('오늘','B_DT'),('날씨','B_DT')
# -> 외부 api(날씨)
