#The purpose of this file is make predictions according model trained
#in 'Train recommender model' file with the
#dataset https://www.kaggle.com/grouplens/movielens-20m-dataset
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle

#Defines the cosine similarity function
def cosine_similarity(u, v):

    distance = 0.0
    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(np.square(u)))
    norm_v = np.sqrt(np.sum(np.square(v)))
    cosine_similarity = np.divide(dot,np.multiply(norm_u,norm_v))
    return cosine_similarity

#Load dataset and pretrained model
df = pd.read_csv("movies_small_data.csv")
model = load_model("recommender_model.h5")

#Load movie ID to title dictionary
with open('movieID2title.json', 'rb') as json:
    movieID2title = pickle.load(json)

n_total_rates = df.shape[0]
unique_movies = pd.unique(df.movie_idx)
unique_users = pd.unique(df.userId)
n_uniq_movies = len(unique_movies)
n_uniq_users = len(unique_users)
mu = df.rating.mean()

#Print examples of users an its predicted ratings for movies
n_predictions = 5
print("===========================================================================")
for _ in range(n_predictions):
    selected_user = np.random.randint(0, n_uniq_users)
    selected_movie_idx = np.random.randint(0, n_uniq_movies)
    
    selected_example = [[selected_user], [selected_movie_idx]]
    prediction = float(model.predict(selected_example)) + mu

    selected_movie_id = df.loc[selected_movie_idx, 'movieId']

    print("For user {}, the prediction for the movie {} rating is {:.2f}".format(selected_user, movieID2title[selected_movie_id], prediction))

print("===========================================================================")

###########################################################################################
#The next section of code select a random user and print the most n
#recommended movies for him

n_favs=10
#Select random user
selected_user = np.random.randint(0, n_uniq_users)
selected_user = np.repeat(selected_user, n_uniq_movies)

#Make the rate prediction for all movies given a specific user
favourites_selected = [selected_user, unique_movies]
favourites_prediction = (model.predict(favourites_selected) + mu).reshape(-1).astype(np.float32)
favourites_array = np.vstack((favourites_prediction, unique_movies))

#Sort by ratings column and take the n elements with the max rate
favourites_df = pd.DataFrame(data=favourites_array.T, columns=['rating', 'movie_idx'])
favourites_df = favourites_df.sort_values(by=['rating'], ascending=False)
final_favourites = favourites_df.iloc[:n_favs, :]

#Print the five most recommended movies for a random user
print("===========================================================================")
print("The most {} recommended movies for user {} are:".format(n_favs, selected_user[0]))
for i in range(n_favs):
    movie_idx =  final_favourites.iloc[i, :].movie_idx
    decoded_movie_id = df.loc[movie_idx, 'movieId']
    rate = final_favourites.iloc[i, :].rating
    print("Movie: {} with rate: {:.2f}".format(movieID2title[decoded_movie_id], rate))

print("===========================================================================")

########################################################################################
#This fragment of code calculate the users with the most similar ratings and 
#Compares de rating predictions for n movies 

#Load the trained weights of embbeding user matrix
u_embedding_weights = model.get_layer('embedding_1').get_weights()[0]

#Randomly select the user id and extract
similarity_user_id = np.random.randint(0,n_uniq_users)
user_vector = u_embedding_weights[similarity_user_id]

#Calculate the cosine similarity of the selected vector front all
#vectors in the embedding users matrix
similarity = []
for i in range(0, n_uniq_users):
    u = user_vector
    v = u_embedding_weights[i]
    similarity.append(cosine_similarity(u,v))

similarity_array = np.vstack((similarity, unique_users))

#Sort by similarity column and take the user with the most similarity
similarity_df = pd.DataFrame(data=similarity_array.T, columns=['similarity', 'userId'])
similarity_df = similarity_df.sort_values(by=['similarity'], ascending=False)
most_similar = similarity_df.iloc[:2, :]
#Pick most similar avoiding itself
most_similar = most_similar.iloc[-1, :]
most_similar_id = int(most_similar['userId'])

print("===========================================================================")
print("The most similar user of {} is the user: {}".format(similarity_user_id, most_similar_id))
print("The predicted rate of user {} and user {} for the next movies:".format(similarity_user_id, most_similar_id))

#Print and compare n predictions for most similar users selected
for i in range(5):
    selected_movie_idx = np.random.randint(0, n_uniq_movies)
    decoded_movie_id = df.loc[selected_movie_idx, 'movie_idx']
    
    example_user1 = [[similarity_user_id], [selected_movie_idx]]
    prediction1 = float(model.predict(example_user1)) + mu
    example_user2 = [[most_similar_id], [selected_movie_idx]]
    prediction2 = float(model.predict(example_user2)) + mu

    print("Movie: {} with user {} rate: {:.2f} and user {} rate: {:.2f}".format(
        movieID2title[decoded_movie_id], similarity_user_id, prediction1, most_similar_id, prediction2))
print("===========================================================================")

########################################################################################
#This fragment of code calculate the most similar movies accordin its embeddings

#Load the trained weights of embbeding movies matrix
m_embedding_weights = model.get_layer('embedding_2').get_weights()[0]

#Randomly select the user id and extract
similarity_movie_idx = np.random.randint(0,n_uniq_movies)
user_vector = u_embedding_weights[similarity_movie_idx]

#Calculate the cosine similarity of the selected vector front all
#vectors in the embedding movies matrix
movies_similarity = []
for i in range(0, n_uniq_movies):
    u = user_vector
    v = u_embedding_weights[i]
    movies_similarity.append(cosine_similarity(u,v))

movies_similarity_array = np.vstack((movies_similarity, unique_movies))

#Sort by similarity column and take the movies with the most similarity
movies_similarity_df = pd.DataFrame(data=movies_similarity_array.T, columns=['similarity', 'movie_idx'])
movies_similarity_df  = movies_similarity_df .sort_values(by=['similarity'], ascending=False)
most_similar_movie = movies_similarity_df.iloc[:2, :]
#Pick most similar avoiding itself
most_similar_movie = most_similar_movie.iloc[-1, :]
most_similar_movie_idx = int(most_similar_movie['movie_idx'])

decoded_movie_id = df.loc[similarity_movie_idx, 'movie_idx']
most_similar_movie_id = df.loc[most_similar_movie_idx, 'movie_idx']


print("===========================================================================")
print("According the ratings, the most similar to movie {} is the movie: {}".format(movieID2title[most_similar_movie_id], movieID2title[decoded_movie_id]))
