from sklearn.decomposition import PCA
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

X = df.drop('Year', axis=1)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, df['GDP']], axis=1)
PCA_df['GDP'] = LabelEncoder().fit_transform(PCA_df['GDP'])
print(PCA_df.head())
