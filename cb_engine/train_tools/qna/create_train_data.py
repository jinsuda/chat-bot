from configs.DatabaseConfig import *

import pymysql

print(DB_HOST)

import pandas as pd

db = None

try:
    # DB 연결
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        port=DB_PORT,
        charset="utf8",
    )

    # 내일 오전 10시에 탕수육 주문 가능할까요?
    # 의도 : 음식 주문
    # 개체명 : 내일(Date), 오전10시(Time), 탕수육(Food)

    # intent : 의도
    # ner : 개체명
    # query : 질문
    # answer : 답변
    # answer_sub_url : 답변 url
    sql = """
      CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
      `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      `answer_sub_url` VARCHAR(2048) NULL,
      PRIMARY KEY (`id`))
    ENGINE = InnoDB DEFAULT CHARSET=utf8
    """

    with db.cursor() as cursor:
        cursor.execute(sql)


except Exception as e:
    print("실패")

finally:
    if db is not None:
        db.close()
