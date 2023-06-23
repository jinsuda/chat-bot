import pymysql
import openpyxl
from configs.DatabaseConfig import *


# 학습데이터 초기화
def all_clear_train_data(db):
    # 기존 데이터 전체 삭제
    # delete : 내용 삭제
    # drop : 테이블 삭제

    sql = """
        delete from chatbot_train_data
    """
    with db.cursor() as cursor:
        cursor.execute(sql)

    # auto_increment 초기화
    sql = """
        alter table chatbot_train_data AUTO_INCREMENT=1
    """
    with db.cursor() as cursor:
        cursor.execute(sql)


# 데이터 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_sub_url = xls_row

    sql = """
        insert chatbot_train_data(intent,ner,query,answer,answer_sub_url)
        values('%s','%s','%s','%s','%s')
    """ % (
        intent.value,
        ner.value,
        query.value,
        answer.value,
        answer_sub_url.value,
    )

    sql = sql.replace("'None'", "null")

    with db.cursor() as cursor:
        cursor.execute(sql)
        db.commit()


db = None
train_file = "cb_engine/train_tools/qna/answer.xlsx"
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

    # 초기화 함수
    all_clear_train_data(db)

    # insert
    wb = openpyxl.load_workbook(train_file)
    sheet = wb["Sheet1"]
    for row in sheet.iter_rows(min_row=2):
        insert_data(db, row)
    print(insert_data)
    wb.close()

except Exception as e:
    print("실패", str(e))

finally:
    if db is not None:
        db.close()
