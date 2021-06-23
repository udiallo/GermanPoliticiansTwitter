from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Load data
df1 = pd.read_pickle("data/german_politicans_tweets_with_features.pkl")

# Create "partei" label
users_gruen = ["Die_Gruenen", "ob_palmer", "ABaerbock", "sven_giegold", "MiKellner", "jamila_anna"]
users_spde =                   ["spdde",  "Karl_Lauterbach", "OlafScholz", "NowaboFM", "HeikoMaas", "Doro_Martin"]
users_linke = ["dieLinke", "Janine_Wissler", "JoergSchindler", "SusanneHennig", "Si_Wagenknecht", "DietmarBartsch"]
users_cdu = ["CDU", "jensspahn", "CDUMerkel", "Markus_Soeder", "HGMaassen", "PaulZiemiak"]
users_afd = ["AfD", "BjoernHoecke", "Tino_Chrupalla", "Alice_Weidel", "Beatrix_vStorch", "M_HarderKuehnel"]
users_fdp = ["c_lindner", "Wissing", "johannesvogel", "moritzkoerner", "MarcoBuschmann"] #fdp

df1["partei"] = ""
for index, row in df1.iterrows():
    if row["user"] in users_gruen:
       df1.at[index, "partei"] = "diegruenen"
    elif row["user"] in users_spde:
       df1.at[index, "partei"] = "spd"
    elif row["user"] in users_linke:
       df1.at[index, "partei"] = "dielinke"
    elif row["user"] in users_cdu:
       df1.at[index, "partei"] = "cdu"
    elif row["user"] in users_afd:
       df1.at[index, "partei"] = "afd"
    elif row["user"] in users_fdp:
       df1.at[index, "partei"] = "fdp"

# Exclude jensspahn and merkel as there are only very few samples available
df1 = df1[df1.user != "jensspahn"]
df1 = df1[df1.user != "CDUMerkel"]

# Split into test and train sets
df_train, df_test = train_test_split(df1, test_size=0.1)

# select features, set number of labels, get train set

feature_selection = "text_processed"
select = ["user", "partei"] # "text", "text_lemmatized",
for t in select:
    labels = t  # "partei"
    number_of_label = df_train[labels].nunique()
    print("Number of labels: ")
    print(number_of_label)
    Y_train = df_train[labels]
    X_train = df_train[feature_selection]

    # Convert to tfidf representation
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train) # text_lemmatized
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train classifier
    clf = MultinomialNB().fit(X_train_tfidf, Y_train)

    # Test classifier
    X_test = list(df_test[feature_selection])
    Y_test = df_test[labels]
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    print(clf.score(X_new_tfidf, Y_test))
    print(classification_report(Y_test, predicted, target_names=clf.classes_))

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_confusion_matrix(clf, X_new_tfidf, Y_test, ax=ax, xticks_rotation="vertical")
    plt.savefig("plots/confusion_mat_NB_" + "target_" + labels + "_" + feature_selection + ".png")
    # plt.show()

    # Use SVM model
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])
    text_clf.fit(X_train, Y_train)
    predicted = text_clf.predict(X_test)
    print(classification_report(Y_test, predicted, target_names=text_clf.classes_))
    print(np.mean(predicted == Y_test))

    # Grid search for parameter tuning
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    gs_clf.fit(X_train[:1000], Y_train[:1000])

    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_confusion_matrix(text_clf[2], X_new_tfidf, Y_test, ax=ax, xticks_rotation="vertical") # , normalize="true"
    plt.savefig("plots/confusion_mat_SVM_"  + "target_" + labels + "_" + feature_selection + ".png")
    # plt.show()
