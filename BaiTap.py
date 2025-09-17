import numpy as np
import re

# Du lieu
documents=[ "Machine Learning cơ bản",
    "Machine Learning rất vui!",
    "Học Machine Learning khi rảnh và không rảnh."]
# Xu li du lieu
def XuLyDuLieu(text):
    text=text.lower()
    #text=re.sub(r'[^a-zA-Z0-9\s]','',text,flags=re.UNICODE)
    text=re.sub(r'[^\w\s]','',text,flags=re.UNICODE)
    tokens=text.split()
    return tokens
du_lieu_da_xu_ly=[XuLyDuLieu(text) for text in documents]
all_tokens=[token for doc in du_lieu_da_xu_ly for token in doc]
vocab=sorted(list(set(all_tokens)))

print("Danh Sách từ vựng:",vocab)
#Tinh TF cho 1 cau

def TinhTF(cac_tu_trong_cau,tu_vung):
    tf_vector=[]
    tong_so_tu=len(cac_tu_trong_cau)
    for tu in tu_vung:
        dem=cac_tu_trong_cau.count(tu)
        tf_vector.append(dem/tong_so_tu)
    return tf_vector

# Tinh TF cho tat ca cau
ma_tran_tf=[TinhTF(cau,vocab) for cau in du_lieu_da_xu_ly]
for dong in ma_tran_tf:
    print(dong)

import math

# Tinh IDF
def Tinh_IDF(tu_vung, du_lieu):
    N = len(du_lieu)
    idf_vector = []
    for tu in tu_vung:
        so_cau_chua_tu = sum(1 for cau in du_lieu if tu in cau)
        idf = math.log(N / (1 + so_cau_chua_tu))
        idf_vector.append(idf)
    return idf_vector

idf_vector = Tinh_IDF(vocab, du_lieu_da_xu_ly)


# Tính TF-IDF cho từng câu
ma_tran_tfidf = []
for tf_vec in ma_tran_tf:
    tfidf_vec = [tf * idf for tf, idf in zip(tf_vec, idf_vector)]
    ma_tran_tfidf.append(tfidf_vec)

print("Ma trận TF-IDF:")
for dong in ma_tran_tfidf:
    print(dong)

