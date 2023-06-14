# 챗봇에서 사용할 단어 사전 만들기
# corpus.txt -> 말뭉치 데이터
# 0 : 인사
# 1 : 욕설
# 2 : 주문
# 3 : 예약
# 4 : 기타

from utils.preprocessing import Preprocessing
from tensorflow.keras import preprocessing
import pickle


# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, "r", encoding="utf8") as f:
        # 줄별로 먼저 나누고 탭으로 나눔
        data = [line.split("\t") for line in f.read().splitlines()]
    return data


corpus_data = read_corpus_data("cb_engine/train_tools/dict/corpus.txt")

# 말뭉치 데이터에서 키워드만 추출해서 사전 만들기
p = Preprocessing(userdic="cb_engine/utils/user_dic.tsv")
dict = []

for c in corpus_data:
    pos = p.pos(c[1])
    for k in pos:
        # 품사태그 없이 토큰만
        dict.append(k[0])

# 사전에서 사용될 word2index생성
tokenizer = preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index
# 단어 사전 파일로 만들기
f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()
