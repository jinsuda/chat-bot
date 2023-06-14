from utils.preprocessing import Preprocessing
from models.intent.intentModel import IntentModel

p = Preprocessing(
    word2index_dic="cb_engine/train_tools/dict/chatbot_dict.bin",
    userdic="cb_engine/utils/user_dic.tsv",
)

intent = IntentModel(model_name="cb_engine/models/intent/intent_model.h5", preprocess=p)
query = "1월 5일 대한항공 항공권 예매 하고싶어"
predict = intent.predict_class(query)
print(predict)
