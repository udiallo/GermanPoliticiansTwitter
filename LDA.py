import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)

df = pd.read_pickle("data/german_politicans_tweets_with_features.pkl")
number_of_label = df['user'].nunique()
col_select = "text"
d = []
size_train_set = len(df.index) / 2
test_set = []
for index, row in df.iterrows():
    if index > size_train_set:
        test_set.append((row[col_select], row["user"]))
    else:
        d.append((row[col_select], row["user"]))
X_train = [x[0] for x in d] # Text
X_train = np.zeros((len(X_train), len(X_train[0])))
Y_train = [y[1] for y in d] # Label

X_test = [x[0] for x in test_set] # Text
X_test = np.zeros((len(X_test), len(X_test[0])))
Y_test = [y[1] for y in test_set] # Label

processed_docs = X_train
dictionary = gensim.corpora.Dictionary()

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)


'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

'''
Preview BOW for our sample preprocessed document
'''
document_num = 20
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0],
                                                     dictionary[bow_doc_x[i][0]],
                                                     bow_doc_x[i][1]))
'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
# TODO
lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = 8,
                                   id2word = dictionary,
                                   passes = 10,
                                   workers = 2)
'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

num = 100
unseen_document = X_test[num] #newsgroups_test.data[num]
print(unseen_document)

# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(unseen_document)

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))