from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


df = pd.read_pickle("data/german_politicans_tweets.pkl")
names = ["CDUMerkel", "c_lindner", "jensspahn"]

print(df["user"].nunique)