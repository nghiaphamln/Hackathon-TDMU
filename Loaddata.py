import pandas as pd
fileName = 'Vietnam-Macroeconomic-Data.xls'
df = pd.read_excel(fileName)
print(df)

# convert row to col
df1 = df.T
print(df1)