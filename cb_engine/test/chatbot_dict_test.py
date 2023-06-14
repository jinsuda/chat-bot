import pickle
from utils.preprocessing import Preprocessing

f = open("cb_engine/train_tools/dict/chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

sentence = "내일 오전 10시에 대한항공 비행기표 예매 할게요"

p = Preprocessing(userdic="cb_engine/utils/user_dic.tsv")

pos = p.pos(sentence=sentence)

keyword = p.get_keyword(pos, without_tag=True)

# 토큰 -> 인덱스 (우리가 만든 단어사전에서)
for word in keyword:
    try:
        print(word, word_index[word])
    except KeyError:
        print(word, word_index["OOV"])
