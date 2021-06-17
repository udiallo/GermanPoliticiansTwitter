from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

import matplotlib.pyplot as plt


df = pd.read_pickle("data/german_politicans_tweets_with_features.pkl")
names = ["CDUMerkel", "c_lindner", "jensspahn"]
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df["text"])
print(X_train_counts.shape)
print(df.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

feature_selection = "all_features"
number_of_label = df['user'].nunique()
d = []
for index, row in df.iterrows():
    d.append((row[feature_selection], row["user"]))

X_train = [x[0] for x in d] # Text
X_train = np.zeros((len(X_train), len(X_train[0])))


Y_train = [y[1] for y in d] # Label

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
encoded_Y = Y_train

Y_train = df["user"]
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, encoded_Y)

docs_new = ['God is love', 'OpenGL on the GPU is fast', df["text_processed"][900]]
docs_new = list(df["text_processed"])
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

print(clf.score(X_new_tfidf, encoded_Y))


count = 0
count2 = 0
for doc, category in zip(docs_new, predicted):
    # print('%r => %s' % (doc, encoder.inverse_transform([category])))
    if df["user"][count] == category:
        # print("Prediction true")
        count2 += 1
        pass
    count += 1

print(count2/count)

from sklearn.metrics import classification_report
print(classification_report(encoded_Y, predicted, target_names=clf.classes_))

from sklearn.metrics import plot_confusion_matrix

# plt.figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(15, 10))
plot_confusion_matrix(clf, X_new_tfidf, encoded_Y, ax=ax)
plt.savefig("plots/confusion_mat_pol.png")
plt.show()
