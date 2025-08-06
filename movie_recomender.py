import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression # used for k-fold testing and training
from sklearn.metrics import precision_score, recall_score   # used for k-fold scoring
from sklearn.impute import SimpleImputer

##  used for some data visualzation
import seaborn as sns
import matplotlib.pyplot as plt
##

import numpy as np


df = pd.read_csv('n_movies.csv')

v = []
titles = []

twenty = int(len(df) * 0.5)
random.seed(42)

v = random.sample(range(len(df)), twenty)  # unique index values
titles = df['title'].iloc[v].tolist()

watched = []
ct = 0
for i in range(len(df)):
    if i in v:
        watched.append(1)
        ct += 1
    else:
        watched.append(0)
df["watched"] = watched
df = df.dropna(subset=['rating', 'duration', 'watched'])
df.to_csv('movies_with_predictions.csv', index=False)

# *****step 1*****: make sure duration column is an int
df['duration'] = df['duration'].str.replace(' min', '')
# df = df.dropna(subset=['duration'])
df['duration'] = df['duration'].astype(int) # make the float into an int
df.to_csv('movies_with_predictions.csv', index=False)

# *****step 2*****: extract my input columns into the variable X
X = df[["rating","duration"]]

# *****step 3*****: extract my output column into the variable y
y = df["watched"]

# *****step 4*****: split data into training and testing sets - 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # *****step 4 subset*****: training using k-fold cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

imputer = SimpleImputer(strategy='mean')

precision_scores = []
recall_scores = []
for train_index, test_index in kf.split(X_train, y_train):
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

    # Fit imputer on training fold and transform both train and test folds
    X_fold_train = imputer.fit_transform(X_fold_train)
    X_fold_test = imputer.transform(X_fold_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_fold_train, y_fold_train)

    y_pred = model.predict(X_fold_test)

    precision = precision_score(y_fold_test, y_pred)
    recall = recall_score(y_fold_test, y_pred)

    precision_scores.append(precision)
    recall_scores.append(recall)

# print("Precision scores for each fold:", precision_scores)
# print("Recall scores for each fold:", recall_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
# print("Average Precision:", avg_precision)  # average is ~52%   -> ~48% for false positives, said it was watched when it wasn't
# print("Average Recall:", avg_recall)    # average is ~59%
# not ideal. this means the model is better at finding watched movies than predicting them

# *****step 5*****: train the model using the training set
# Train the model on full training data
# print(X_train.isna().sum())
# print(X_test.isna().sum())
# *****step 6*****: train final model
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)

# *****step 7*****: Predict on the 20% test set
y_test_pred = final_model.predict(X_test)

# *****step 8*****: model performance
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
print("Test Precision:", precision) # approximately ~51% - about half of the time it says it was watched, it was actually watched
print("Test Recall:", recall)   # approximately ~37%


# Now recommend the movies with the highest "watched" probability from the precision variable
# Prepare full feature data (make sure no missing values)
X_full = df[['rating', 'duration']]

# If you used imputer during training, transform data similarly
X_full_imputed = imputer.transform(X_full)

# Get probabilities of being watched (class 1)
probs = final_model.predict_proba(X_full_imputed)[:, 1]

# Add probabilities as a new column
df['watched_prob'] = probs

# Recommend top 5 movies with highest watched probability
recommendations = df.sort_values(by='watched_prob', ascending=False).head(5)
df.to_csv("finalized.csv", index=False)

# print("Recommended movies to watch:")
# print(recommendations[['title', 'rating', 'duration', 'watched_prob']])


# y_prob = model.predict_proba(X_test)[:, 1]
# sns.histplot(y_prob, bins=20, kde=True)
# plt.title("Model Prediction Probabilities")
# plt.xlabel("Watched Probability")
# plt.show()
# testing model on whether it will predict if the movie will be watched
new_movie = pd.DataFrame([{
    'title': "Movie Title",
    'year': 2024,
    'certificate': 'PG-13',
    'duration': 110,
    'genre': 'Action',
    'rating': 7.8,
    'description': "An action-packed sci-fi adventure.",
    'stars': "['Some producer', '| ', '    Stars:', 'Actor One, ', 'Actor Two, ']",                # or number of main actors?
    'votes': 15000,
    'watched': None,             # we don't know since a new prediction
    'watched_prob': None         # what we are trying to get
}])
feature_cols = ['rating','duration']
X_new = new_movie[feature_cols]
prob = model.predict_proba(X_new)[0, 1]
print(f"Predicted watched probability: {prob:.2%}") # {prob:.2%} trucates percentage values after the second decimal place
print(f"Which means that based on the model and what it is testing on, there is a {prob:.2%} chance that the movie will be watched by the user")


