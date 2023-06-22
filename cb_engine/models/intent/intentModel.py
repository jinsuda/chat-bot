import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np

# 만들어놓은 h5파일 이용해서 의도 분류


class IntentModel:
    def __init__(self, model_name, preprocess):
        # 클래스별 레이블
        self.label = {0: "인사", 1: "욕설", 2: "정보", 3: "예약", 4: "기타", 5: "맛집"}

        # 분류 모델 불러오기
        self.model = load_model(model_name)

        # 전처리 삭제
        self.p = preprocess

    # 의도 클래스 예측
    def predict_class(self, query):
        # 형태소 분석
        pos = self.p.pos(query)
        keyword = self.p.get_keyword(pos, without_tag=True)
        sequence = [self.p.get_wordindex_sequence(keyword)]

        pad_seq = preprocessing.sequence.pad_sequences(
            sequence, maxlen=15, padding="post"
        )

        pred = self.model.predict(pad_seq)
        # 소프트맥스 함수로 된 배열 리턴
        pred_class = np.argmax(pred, axis=1)
        return pred_class[0]
