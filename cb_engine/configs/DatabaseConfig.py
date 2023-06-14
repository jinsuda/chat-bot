# DB연결 관련한 글로벌 변수

DB_HOST = '127.0.0.1'
DB_USER = 'chatbot'
DB_PASSWORD = 'root'
DB_NAME = 'chatbot'
DB_PORT = 3306

def DatabaseConfig():
    global DB_HOST,DB_USER,DB_PASSWORD,DB_NAME,DB_PORT