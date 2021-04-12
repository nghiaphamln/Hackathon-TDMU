import itertools
import math
from sklearn.metrics import mean_squared_error
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
data = pd.DataFrame({'Year': Year[1:], 'GDP': GDP[1:], 'Unemployment rate': Urate[1:]})
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
new_data = []
for i in X_pred:
    val = round(regression.predict(np.array([[i]]))[0], 2)
    new_data.append(val)
# print(new_data)
clean_Urate = new_data + list(data['Unemployment rate'].iloc[10:])

clean_data = pd.DataFrame({'Year': Year[1:], 'GDP': GDP[1:], 'Unemployment rate': clean_Urate})
clean_data.to_csv('data.csv', index=None)

print("\n----ĐIỀN LẠI CÁC GIÁ TRỊ BỊ THIẾU KHUYẾT----\n")
print(clean_data)

list_GDP = [clean_data['GDP'].min().round(2), clean_data['GDP'].mean().round(2), clean_data['GDP'].median().round(2),
            clean_data['GDP'].max().round(2)]
list_Urate = [clean_data['Unemployment rate'].min().round(2), clean_data['Unemployment rate'].mean().round(2),
              clean_data['Unemployment rate'].median().round(2), clean_data['Unemployment rate'].max().round(2)]

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

df_cor = pd.DataFrame(data=clean_data, columns=['GDP', "Unemployment rate"])

print("\n--------------Độ tương quan---------------\n")
print(df_cor.corr().round(2))

# biểu đồ thể hiện độ tương quan
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
colormap = sns.diverging_palette(145, 10, as_cmap=True)
sns.heatmap(df_cor.corr(), cmap=colormap, annot=True)
plt.xticks(range(df_cor.shape[1]), df_cor.columns, fontsize=10, rotation=45)
plt.yticks(range(df_cor.shape[1]), df_cor.columns, fontsize=10, rotation=0)
plt.title('Đồ thị Heatmap\n', fontsize=20)
plt.show()

X = df_cor.iloc[:, :1].values
y = df_cor.iloc[:, 1].values
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred1 = lin_reg.predict(X_test)
# print(y_pred1)
linear_mse = mean_squared_error(y_test, y_pred1)
linear_rmse = math.sqrt(mean_squared_error(y_test, y_pred1))

# Visualisng the linear regression model results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('GDP')
plt.ylabel('Unemployment rate')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
y_pred2 = lin_reg2.predict(poly_reg.fit_transform(np.array(X_test)))
# print(y_pred)
poly_mse = mean_squared_error(y_test, y_pred2)
poly_rmse = math.sqrt(mean_squared_error(y_test, y_pred2))
# Visualising the pollynomial regression model results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('GDP')
plt.ylabel('Unemployment rate')
plt.show()

print("\n------SO SÁNH MÔ HÌNH HỒI QUY-------------\n")
print("\n-------------Linear Regression------------\n")
print("Sai số toàn phương trung bình: ", round(linear_mse, 2))
print("Độ lệch chuẩn sai số: ", round(linear_rmse, 2))
X_test = X_test.tolist()
X_test = sum(X_test, [])
print("\nKết quả dự đoán")
print("GPD:".ljust(25), end="")
for i in X_test:
    print(str(round(i, 2)).ljust(10), end=" ")
print("\nUnemplement rate:".ljust(25), end=" ")
for i in y_pred1:
    print(str(round(i, 2)).ljust(10), end=" ")

print("\n\n----------Polinomial Regression-----------\n")
print("Sai số toàn phương trung bình: ", round(poly_mse, 2))
print("Độ lệch chuẩn sai số: ", round(poly_rmse, 2))
print("\nKết quả dự đoán")
print("GPD:".ljust(25), end="")
for i in X_test:
    print(str(round(i, 2)).ljust(10), end=" ")
print("\nUnemplement rate:".ljust(25), end=" ")
for i in y_pred2:
    print(str(round(i, 2)).ljust(10), end=" ")
