from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("german_politicans_tweets_with_features.pkl")
names = ["CDUMerkel", "c_lindner", "jensspahn"]

names = ["jensspahn", "CDUMerkel", "c_lindner", "ob_palmer", "HGMaassen", "Si_Wagenknecht", "ABaerbock", "Karl_Lauterbach", "Markus_Soeder", "Die_Gruenen",
                     "fdp", "spdde", "dieLinke", "CDU", "AfDBerlin"]
for name in names:

    df2 = df[df["user"] == name]
    # Import the wordcloud library

    # Join the different processed titles together.
    long_string = ','.join(list(df2['text_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=800, height=400)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()
    wordcloud.to_file("worlcloud_" + name + ".png")