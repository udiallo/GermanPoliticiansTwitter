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
from nltk.stem.snowball import SnowballStemmer
import string
import re
from nltk.corpus import stopwords
from tqdm import tqdm

import empath
stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))
nlp = spacy.load("de_core_news_sm")
lexicon = empath.Empath()

path_to_glove = "vectors.txt"
embeddings_glove = {}
f = open(path_to_glove, "r", encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_glove[word] = coefs
f.close()


# make all text lowercase
def text_lowercase(text):
    return text.lower()
# remove urls
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9_äÄöÖüÜß]+)|([^0-9_äÄöÖüÜßA-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
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
    # text = text.replace("ä", "ae").replace("Ä", "Äe").replace("ö", "oe").replace("Ö", "oe").replace("Ü", "ue").replace("ü", "ue")
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    # text = remove_stopwords(text)
    # text = re.findall(r'(?:\w+)', text, flags=re.UNICODE)
    text = re.sub(r'^.*?:', '', text)
    new_text = []
    sent = re.findall("[A-Za-z0-9_äÄöÖüÜß]+", text)
    line = ''
    text_vec = []
    for words in sent:
        if stop_word_removal == True:
            if len(words) > 1 and words not in stop_words:
                line = line + ' ' + words
                text_vec.append(words)
        else:
            if len(words) > 1:
                line = line + ' ' + words
                text_vec.append(words)
    #     text = tokenize(text)
    #     text = lemmatize(text)
    #     text = ' '.join(text)
    #     text = pos_tag(text) #optional
    return line, text_vec

def clean_text(text, for_embedding=False):
    """
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - convert all whitespaces (tabs etc.) to single wspace
        if not for embedding (but e.g. tdf-idf):
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    if for_embedding:
        # Keep punctuation
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text)
    words_tokens_lower = [word.lower() for word in word_tokens]

    if for_embedding:
        # no stemming, lowering and punctuation / stop words removal
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)
    return text_clean


stop_word_removal = True
# df2 = pd.DataFrame(columns=["user", "id", "text", "entities", "retweet_count", "favorite_count", "retweeted", "spacy_feature"])

df = pd.read_pickle("data/german_politicans_tweets.pkl")
df = df[df["retweeted"] == False]
df = df.reset_index(drop=True)
df2 = df.copy()
df2["text_processed"] = ""
df2["spacy_feature"] = np.empty((len(df), 0)).tolist()
df2["glove_feature"] = np.empty((len(df), 0)).tolist()
df2["all_features"] = np.empty((len(df), 0)).tolist()
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text = row["text"]
    new_text = clean_text(text, for_embedding=True)
    text_string, text_vec = preprocessing(text)
    df2.at[index, "text_processed"] = text_string
    doc = nlp(new_text)
    spacy_feature = doc.vector
    df2.at[index, "spacy_feature"] = spacy_feature
    glove_average_embedding = [0] * 300
    for word in text_vec:
        if word in embeddings_glove:
            glove_for_word = embeddings_glove[word]
            glove_average_embedding = [a + b for a, b in zip(glove_average_embedding, glove_for_word)]

    df2.at[index, "glove_feature"] = glove_average_embedding


    df2.at[index, "all_features"] = np.concatenate((spacy_feature, glove_average_embedding ), axis=None)

df2 = df2[df2["retweeted"] == False]
df2.to_pickle("data/german_politicans_tweets_with_features.pkl", protocol=4)

