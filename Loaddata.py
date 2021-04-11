import math

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

fileName = 'Vietnam-Macroeconomic-Data.xls'
df = pd.read_excel(fileName)
Year, GDP, Urate = [], [], []
list_data = []
for row in df:
    Year.append(row)
for row in df.iloc[0]:
    GDP.append(row)
for row in df.iloc[1]:
    if row == 'no data': row = np.nan
    Urate.append(row)
list_data = [Year, GDP, Urate]
data = pd.DataFrame({'Year' : Year[1:], 'GDP' : GDP[1:], 'Unemployment rate' : Urate[1:]})
X = (data['GDP'].iloc[10:]).values
X_train = []
for i in X:
    X_train.append([i])
# print(X_train)
y_train = (data['Unemployment rate'].iloc[10:]).values
# print(y_train)
X_pred = (data['GDP'].iloc[:10]).values

# print(X_pred)
regression = LinearRegression()
regression.fit(X_train, y_train)
new_data=[]
for i in X_pred:
    val=round(regression.predict(np.array([[i]]))[0],2)
    new_data.append(val)
# print(new_data)
clean_Urate = new_data + list(data['Unemployment rate'].iloc[10:])

clean_data = pd.DataFrame({'Year' : Year[1:], 'GDP' : GDP[1:], 'Unemployment rate': clean_Urate})
clean_data.to_csv('data.csv', index = None)

print("\n----ĐIỀN LẠI CÁC GIÁ TRỊ BỊ THIẾU KHUYẾT----\n")
print(clean_data)


list_GDP = [clean_data['GDP'].min().round(2), clean_data['GDP'].mean().round(2), clean_data['GDP'].median().round(2), clean_data['GDP'].max().round(2)]
list_Urate = [clean_data['Unemployment rate'].min().round(2), clean_data['Unemployment rate'].mean().round(2), clean_data['Unemployment rate'].median().round(2), clean_data['Unemployment rate'].max().round(2)]

# print(list_GDP)
# print(list_Urate)



print("\n--------------------------------------------\n")
print("\n-----------THỐNG KÊ BỘ DỮ LIỆU--------------\n")
print("\n-----------GDP, current prices--------------\n")
print("Giá trị nhỏ nhất: ".ljust(40) + str(list_GDP[0]))
print("Giá trị trung bình: ".ljust(39) + str(list_GDP[1]))
print("Giá trị trung vị: ".ljust(39) + str(list_GDP[2]))
print("Giá trị lớn nhất: ".ljust(38) + str(list_GDP[3]))

print("\n------------Unemployment rate---------------\n")
print("Giá trị nhỏ nhất: ".ljust(40) + str(list_Urate[0]))
print("Giá trị trung bình: ".ljust(39) + str(list_Urate[1]))
print("Giá trị trung vị: ".ljust(40) + str(list_Urate[2]))
print("Giá trị lớn nhất: ".ljust(39) + str(list_Urate[3]))

print("\n--------------Độ tương quan---------------\n")
print(clean_data.corr().round(2))
# def update_vals(row, data=data1):
#     if row.GDP == data1['Unemployment rate']:
#         row. = data['n']
#     return row
#
# df.apply(update_vals, axis=1)
# for i in X_pred:
#     (data['Unemployment rate'].iloc[i]).values=new_data[i]
# print((data['Unemployment rate'].iloc[i]).values)

