from utils.preprocessing import Preprocessing

p = Preprocessing(userdic="cb_engine/utils/user_dic.tsv")

pos = p.pos(sentence="내일 오전 10시에 대한항공 비행기표 예약 할게요")

keyword = p.get_keyword(pos, without_tag=True)

# 검증
# sentence를 내일 오전 10시에 탕수육 주문할게요 로 했으니까
# keyword로 뭘 리턴받을지 미리 예상
# '내일', '오전', '10시

# if(keyword =='내일', '오전', '10시'):
#     # 서버구동
# else :
#     # 중단

print(keyword)

keyword = p.get_keyword(pos, without_tag=False)
print(keyword)
