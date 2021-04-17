import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('Quote.csv', encoding='windows-1251')

data = df['Quote'].tolist()
# train test 8:2
train, test = train_test_split(data, test_size=0.2)


# instantiate the vectorizer object
countvectorizer = CountVectorizer(analyzer='word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')


# convert th documents into a matrix
count_wm = countvectorizer.fit_transform(train)
tfidf_wm = tfidfvectorizer.fit_transform(train)

count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_countvect = pd.DataFrame(data=count_wm.toarray(), columns=count_tokens)
df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
    print("Count Vectorizer\n")
    print(df_countvect)
    print("\nTD-IDF Vectorizer\n")
    print(df_tfidfvect)
