import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
# import de_core_news_sm
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# # from textblob import TextBlob
import string
import re
from nltk.corpus import stopwords
from tqdm import tqdm

import empath
stop_words = set(stopwords.words('german'))
nlp = spacy.load("de_core_news_sm")
lexicon = empath.Empath()


# make all text lowercase
def text_lowercase(text):
    return text.lower()
# remove urls
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text
# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result
# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text
# lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text
# POS tag
def pos_tag(text):
    return nltk.pos_tag(nltk.word_tokenize(text))
# chain all functions
def preprocessing(text):
    text = text.replace("ä", "ae").replace("Ä", "Äe").replace("ö", "oe").replace("Ö", "oe").replace("Ü", "ue").replace("ü", "ue")
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    # text = remove_stopwords(text)
    # text = re.findall(r'(?:\w+)', text, flags=re.UNICODE)
    text = re.sub(r'^.*?:', '', text)
    new_text = []
    sent = re.findall("[A-Za-z]+", text)
    line = ''
    for words in sent:
        if stop_word_removal == True:
            if len(words) > 1 and words not in stop_words:
                line = line + ' ' + words
        else:
            if len(words) > 1:
                line = line + ' ' + words
    #     text = tokenize(text)
    #     text = lemmatize(text)
    #     text = ' '.join(text)
    #     text = pos_tag(text) #optional
    return text, sent


stop_word_removal = True
# df2 = pd.DataFrame(columns=["user", "id", "text", "entities", "retweet_count", "favorite_count", "retweeted", "spacy_feature"])

df = pd.read_pickle("german_politicans_tweets.pkl")
df = df[df["retweeted"] == False]
df = df.reset_index(drop=True)
df2 = df.copy()
df2["text_processed"] = ""
df2["spacy_feature"] = np.empty((len(df), 0)).tolist()
df2["empath_feature"] = np.empty((len(df), 0)).tolist()
df2["all_features"] = np.empty((len(df), 0)).tolist()
for index, row in tqdm(df.iterrows(), "Loop through tweets"):
    text = row["text"]
    new_text, new_text_list = preprocessing(text)
    df2.at[index, "text_processed"] = new_text
    doc = nlp(new_text)
    spacy_feature = doc.vector
    df2.at[index, "spacy_feature"] = spacy_feature

    empath_dict = lexicon.analyze(new_text)
    # if index % 10 == 0:
    #     print(empath_dict)
    empath_features = list(empath_dict.values())

    df2.at[index, "empath_feature"] = empath_features

    df2.at[index, "all_features"] = np.concatenate((spacy_feature, empath_features), axis=None)

df2 = df2[df2["retweeted"] == False]
df2.to_pickle("german_politicans_tweets_with_features.pkl")

