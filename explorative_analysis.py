from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
df = pd.read_pickle("data/german_politicans_tweets_with_features.pkl")
names = ["CDUMerkel", "c_lindner", "jensspahn"]

names = ["jensspahn", "CDUMerkel", "c_lindner", "ob_palmer", "HGMaassen", "Si_Wagenknecht", "ABaerbock", "Karl_Lauterbach", "Markus_Soeder", "Die_Gruenen",
                     "fdp", "spdde", "dieLinke", "CDU", "AfDBerlin"]
# for name in tqdm(names):
#
#     df2 = df[df["user"] == name]
#     # Import the wordcloud library
#
#     # Join the different processed titles together.
#     long_string = ','.join(list(df2['text_processed'].values))
#     # Create a WordCloud object
#     wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800, height=400)
#     # Generate a word cloud
#     wordcloud.generate(long_string)
#     # Visualize the word cloud
#     wordcloud.to_image()
#     wordcloud.to_file("wordclouds/worlcloud_" + name + ".png")

# print(df["all_features"][0])

df2 = pd.read_pickle("data/german_politicans_tweets.pkl")
number_of_label = df2['user'].nunique()
my_dict = {}
word_dict = {}
for index, row in df.iterrows():
    if row["user"] in my_dict:
        my_dict[row["user"]] += 1
        word_dict[row["user"]] = word_dict[row["user"]] + " " + row["text_processed"]
    else:
        my_dict[row["user"]] = 1
        word_dict[row["user"]] = row["text_processed"]

# print(my_dict)
df3 = df.loc[df['user'] == "CDUMerkel"]
# print(df3["text"].head())



from collections import Counter
for key in word_dict:
    user = word_dict[key]
    split = user.split()
    coun = Counter(split)
    most_occur = coun.most_common(10)
    print("Most common words for " + key)
    print(most_occur)


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
import operator

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

print(len(df.id.unique()))
print(len(df.id))



vocab = build_vocab(df['text'])


path_to_glove = "vectors.txt"
embeddings_glove = {}
f = open(path_to_glove, "r", encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_glove[word] = coefs
f.close()
embed_glove = embeddings_glove
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)

