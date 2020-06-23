#https://www.kaggle.com/grouplens/movielens-20m-dataset
#This file preprocess a huge 20 Millions movies rating dataset
#To select only the n number of most common users and movies 
#To generate new smaller dataset
import pandas as pd
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--users", default='5000', help="Number of users to select")
parser.add_argument("--movies", default='1000', help="Number of movies to select")
args = parser.parse_args()

#Assign from args number of most common users an movies
n_final_users = int(args.users)
n_final_movies = int(args.movies)

#Load data and drop timestamp column
raw_data = pd.read_csv("rating.csv")
raw_data = raw_data.drop(columns='timestamp')
titles_data = pd.read_csv("movie.csv")
movieID2title = dict(zip(titles_data['movieId'], titles_data['title']))

#Extract the unique id movies and the number of movies and users
raw_data.userId = raw_data.userId - 1
number_movies = len(pd.unique(raw_data.movieId.values))
number_users =  len(pd.unique(raw_data.userId.values))

print("Number of total movies: {0}".format(number_movies))
print("Number of total users: {0}".format(number_users))

ordered_user_id = raw_data.userId.value_counts().sort_values(ascending=False) 
final_users_ids = ordered_user_id.iloc[:n_final_users]
final_users_ids = final_users_ids.index.sort_values()

ordered_movies_id = raw_data.movieId.value_counts().sort_values(ascending=False)
final_movies_ids = ordered_movies_id.iloc[:n_final_movies]
final_movies_ids = final_movies_ids.index.sort_values()

#Make a copy only with the users and movies selected
small_data = raw_data[raw_data.userId.isin(final_users_ids) & raw_data.movieId.isin(final_movies_ids)].copy()
small_data = small_data.reset_index(drop=True)

print("Number of new total movies: {0}".format(len(pd.unique(small_data.movieId.values))))
print("Number of new total users: {0}".format(len(pd.unique(small_data.userId.values))))

#Maps the users and movies to index values 
new_userId = {}
i = 0
for user_id in final_users_ids:
    new_userId[user_id] = i
    i += 1

movie2idx = {}
j = 0
for movie_id in final_movies_ids:
    movie2idx[movie_id] = j
    j += 1

def reorder_movie_idx(row):
    row_processed = int(row.name)
    if row_processed%100000==0:
        percentage = (row_processed/int(small_data.shape[0]))*100
        print("Number of rows processed: {0}".format(row_processed))
        print("Percentage processed: {:.2f}%".format(percentage))
    return movie2idx[row.movieId]

def reorder_userId(row):
    row_processed = int(row.name)
    if row_processed%100000==0:
        percentage = (row_processed/int(small_data.shape[0]))*100
        print("Number of rows processed: {0}".format(row_processed))
        print("Percentage processed: {:.2f}%".format(percentage))
    return new_userId[row.userId]

#Reorder user id starting at ID 0
print("Reorder user ID")
small_data['userId'] = small_data.apply(lambda row: reorder_userId(row), axis=1)
#Create a new column that maps movie ID to index starting at 0
print("Generate movie ID to Index")
small_data['movie_idx'] = small_data.apply(lambda row: reorder_movie_idx(row), axis=1)

print("Number of new matrix movies: {0}".format(len(pd.unique(small_data.movie_idx))))
print("Number of new matrix users: {0}".format(len(pd.unique(small_data.userId))))
print("Size of new matrix: {0}".format(small_data.shape))
print("Maximum user id: {0}".format(small_data.userId.max()))
print("Minimum user id: {0}".format(small_data.userId.min()))
print("Maximum Original movie id: {0}".format(small_data.movieId.max()))
print("Minimum Original movie id: {0}".format(small_data.movieId.min()))
print("Maximum New movie id: {0}".format(small_data.movie_idx.max()))
print("Minimum New movie id: {0}".format(small_data.movie_idx.min()))

#Save data
small_data.to_csv('movies_small_data.csv', index=False)

with open('movieID2title.json', 'wb') as json:
    pickle.dump(movieID2title, json)