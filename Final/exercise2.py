import operator

import pandas as pd
import datetime
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from underthesea.transformer.count import CountVectorizer

df = pd.read_csv("Quote.csv", encoding='windows-1251')

list_STT = []
for i in range(1, len(df.index)+1):
    list_STT.append(i)
df.insert(0, "STT", list_STT, True)

#
list_age = []
list_birth = df["Namsinh"].values.tolist()
for i in range(len(list_birth)):
    list_age.append(datetime.datetime.now().year -  int(list_birth[i].split(", ")[-1]))
# print(list_age)

df.insert(4, "Tuoi", list_age, True)
# print(df.to_string(index=None))

df2 = pd.DataFrame(df, columns=['Tacgia', 'Quote'])
groups = df2.groupby(["Tacgia", "Quote"])
new = groups.first()
new = new.reset_index()

data = {}

for x in new.values.tolist():
    if x[0] in data:
        data[x[0]].append(x[1])
    else:
        data[x[0]] = [x[1]]

for x, y in data.items():
    print("Tác giả:", x)
    print("Quote: ")
    for q in y:
        print(f"\t-{q}")
    print("\n")


df3 = pd.DataFrame(df, columns=['Tacgia', 'Namsinh', 'Tuoi'])
groups = df3.groupby(["Tacgia", "Namsinh", 'Tuoi'])
new = groups.first()
new = new.reset_index()

data = {}
print("\n\n")
for x in new.values.tolist():
    if x[0] in data:
        data[x[0]].append(x[1])
    else:
        data[x[0]] = [x[1]]

for x, y in data.items():
    print("Tác giả:", x)
    print("Năm sinh: ", end="")
    for q in y:
        print(f"{q}")
    print("Tuổi: ", datetime.datetime.now().year -  int(q.split(", ")[-1]))
    print("\n")

list_quote = df["Quote"].values.tolist()
list_author = df["Tacgia"].values.tolist()
def MaxMinWords(list_quote):
    numWords = [len(sentence.split()) for sentence in list_quote]
    return max(numWords), min(numWords)

max_len, min_len = MaxMinWords(list_quote)
for i in list_quote:
    if(len(i.split()) == max_len):
        print("Câu dài nhất: ",i)
        print("Số từ: ", max_len)
    if(len(i.split()) == min_len):
        print("Câu ngắn nhất: ", i)
        print("Số từ: ", min_len)

cv = CountVectorizer()
cv_fit = cv.fit_transform(list_quote)
word_list = cv.get_feature_names();
count_list = cv_fit.toarray().sum(axis=0) #tính tổng theo cột (asix = 0)
tops = dict(zip(word_list, count_list)) #tạo một dictionary với key là tên của từ xuất hiện nhiều nhất và value và số lần đếm được
maxvalue = max(tops.items(), key=operator.itemgetter(1))[0]
print("\nTừ được sử dụng nhiều nhất là : ", maxvalue)
print("Số lần xuất hiện là: ", max(count_list))