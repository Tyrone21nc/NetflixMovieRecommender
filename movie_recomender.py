import pandas as pd
import random

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import ast



df = pd.read_csv('n_movies.csv')
# df.drop(columns = ['year', 'votes'], inplace=True)
# print(df.head())
watched_arr = []
for i in range(len(df)):
    watched_arr.append(0)
watched_movies_tvshows = []

twenty_perct = int(len(df) * 0.2)

# for i in range(twenty_perct):
#     random.seed(42 + i)  # set the seed
#     rand_num = random.randint(0, 9956)
#     watched_movies_tvshows.append(df['title'][rand_num])
#     watched_arr[rand_num] = 1

df["watched"] = watched_arr
df.to_csv('n_movies.csv', index=False)


# drop the 'watched' column from the DataFrame
# df.drop(columns=['certificate', 'duration', 'description', 'stars', 'title'], inplace=True)

for i in range(len(df)):
    if type(df['genre'][i]) != str:
        df = df[df['genre'].notna()]
        df['genre'] = df['genre'].astype(str).str.split(', ')







# mlb = MultiLabelBinarizer()
# genre_encoded = mlb.fit_transform(df['genre'])
# genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
# df_final = pd.concat([genre_df, df[['rating', 'watched']].reset_index(drop=True)], axis=1)
# X = df_final.drop(columns=['watched'])
# y = df_final['watched']





df.to_csv('n_movies.csv', index=False)
X = df



# Split the dataset into training and testing sets




