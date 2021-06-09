import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
df = pd.read_pickle("german_politicans_tweets_with_features.pkl")

feature_selection = "all_features"
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
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# # load dataset
# dataframe = pd.read_csv("iris.csv", header=None)
# dataset = dataframe.values
# X = dataset[:,0:4].astype(float)
# Y = dataset[:,4]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=len(df[feature_selection][9]), activation='relu'))
	model.add(Dense(32, activation='relu')),
	model.add(Dense(3, activation='softmax')),
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))