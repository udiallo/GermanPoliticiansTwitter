import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import keras.optimizers

from sklearn.metrics import confusion_matrix
import seaborn as sn

df = pd.read_pickle("data/german_politicans_tweets_with_features.pkl")

feature_selection = "all_features"
num_classes = df['user'].nunique()
len_features = len(df[feature_selection][9])
d = []
for index, row in df.iterrows():
    vals = np.append(row[feature_selection], row["retweet_count"] )
    vals = np.append(vals, row["favorite_count"])
    d.append((vals, row["user"]))

X = [x[0] for x in d] # Text
X = np.zeros((len(X), len(X[0])))


Y = [y[1] for y in d] # Label


from keras.optimizers import SGD #Sto
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(Y)
# print("before to categorial:", y.shape)
#y = np_utils.to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)



# print(y.shape)
# print(X.shape)
def create_network(dense_size=1024):
    # create model
    model = Sequential()
    model.add(Dense(dense_size, input_dim=(len_features+2), activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))

    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


average_score = []
# for i in range(4):
#     neural_network = create_network()
#     neural_network.fit(X_train, y_train, epochs=50, batch_size=512)
#
#     scores = neural_network.evaluate(X_test, y_test)
#     average_score.append(scores[1])
#
# text_file = open("multiclass_neuralNet_results" + graph_data + ".txt", "w")
# text_file.write("Results for 4 runs:\n")
# text_file.write(str(average_score[0]) + " " + str(average_score[1]) + " " + str(average_score[2]) + " " + str(average_score[3]))
# text_file.write("\n")
# text_file.write("Average result:\n")
# avg = sum(average_score) / 4
# text_file.write(str(avg))
# text_file.close()

layer_size = [8, 32, 64, 128, 256, 512, 1024, 2048]
for l in layer_size:
    neural_network = create_network(l)
    neural_network.fit(X_train, y_train, epochs=50, batch_size=512)

    scores = neural_network.evaluate(X_test, y_test)
    average_score.append(scores[1])

for count, value in enumerate(average_score):
    print("Score: " + str(value) + " with layer size" + str(layer_size[count]))




# y_pred = neural_network.predict(X_test)
# subreddit_list = subreddit_list1_new
# lab = []
# flat_list = [item for sublist in y_test.tolist() for item in sublist]
# labels = set(flat_list)
# for i in labels:
#     lab.append(subreddit_list[i])
#
# y_test_strings = []
# for i in y_test.values:
#     y_test_strings.append(subreddit_list[i])
#
# y_pred_strings = []
# for i in y_pred.values:
#     y_pred_strings.append(subreddit_list[i])
# # matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=lab)
# con_mat = confusion_matrix(y_test_strings, y_pred_strings,  labels=lab)
#
# cmn = con_mat.astype('float') /con_mat.sum(axis=1)[:, np.newaxis]
# array = cmn
#
# df_cm = pd.DataFrame(array, index=lab,
#                      columns=lab)
# plt.figure(figsize=(20, 20))
# #sn.set_palette(sn.color_palette("rocket_r", as_cmap=True))
# cmap = sn.light_palette("#3fdd01", as_cmap=True)
# sn.heatmap(df_cm, annot=True, fmt='.1f', cmap="PiYG", center=0)  #
# plt.savefig("confusion_multiclass" + graph_data + ".png")
# # plt.show()
