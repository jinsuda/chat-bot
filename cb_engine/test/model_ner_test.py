from utils.preprocessing import Preprocessing
from models.ner.NerModel import NerModel
from konlpy.tag import Komoran


p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)

ner = NerModel(model_name="cb_engine/models/ner/ner_model.h5", preprocess=p)

while True:
    query = input("질문:")

    komoran = Komoran()
    print(komoran.pos(query))
    predict = ner.predict(query)
    tag = ner.predict_tag(query)
    # hotel = ner.search_hotel(query)
    # rest = ner.search_rest(query)
    print(predict)
    print(tag)
    # print(hotel)
    # print(rest)
