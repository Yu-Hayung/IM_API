import pandas as pd

# input = .txt 파일
# output = json {[명사 : 빈도, 명사 : 빈도, 명사 : 빈도, 명사 : 빈도, 명사 : 빈도 ...]}



input_data = {'추후', '세액', '신고', '경우', '기', '원천', '원천', '원천', '원천', '원천', '완납', '납', '말', '시점', '의무', '종결', '확정', '과정', '납부', '납세', '징수', '공제', '예', '의무자', '정산', '구분'}


file_name_df = 'C:/Users/yuhay/Desktop/job_word/total_Noun_df.csv'
job_Noun = pd.read_csv(file_name_df, encoding='UTF8')


def Terminology_Extraction(zq_code, input_data):
    code = int(zq_code)
    code_Noun = job_Noun.loc[code]
    code_Noun = code_Noun.values.tolist()
    print('code_Noun >>', code_Noun)
    # print(input_data[0])
    code_Noun = set(code_Noun)
    same = input_data.intersection(code_Noun)
    print(input_data)

    return same



Terminology_Extraction_list = Terminology_Extraction(0, input_data)
print('Terminology_Extraction_list >> ', Terminology_Extraction_list)
