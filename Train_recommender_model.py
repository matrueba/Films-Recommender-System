#https://www.kaggle.com/grouplens/movielens-20m-dataset
#This file create the embeddings of the movies and users matrix
#And train the model to predict movies rating according a specified
#user and movie
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import argparse
from keras.layers import Embedding, Input, Flatten, Concatenate,Dense, Activation
from keras.models import Model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--split", default='0.9', help="Percentage of train dataset (decimal)")
args = parser.parse_args()

#Load data 
df = pd.read_csv("movies_small_data.csv")

N = len(pd.unique(df.userId.values))
M = len(pd.unique(df.movie_idx.values))

split = float(args.split)
len_train = int(len(df)*split)

#Shuffle data and create the train and test splits
df = shuffle(df)
train_df = df.iloc[:len_train]
train_df = train_df.reset_index(drop=True)
test_df = df.iloc[len_train:]
test_df = test_df.reset_index(drop=True)

print("Train dataset size: {0}".format(train_df.shape))
print("Test dataset size: {0}".format(test_df.shape))

#K represents embedding vector size
K = 10
mu = train_df.rating.mean()
epochs = 20

#Generate the final datasets to embbed
x_train = [train_df.userId.values, train_df.movie_idx.values]
y_train = train_df.rating.values - mu
x_test = [test_df.userId.values, test_df.movie_idx.values]
y_test = test_df.rating.values - mu

#Embedding model
u = Input(shape=(1,))
m = Input(shape=(1,))

# Embedding Layers to generate the users and movies embedding matrix
u_embedding = Embedding(N, K)(u)#(N,1,K)
m_embedding = Embedding(M, K)(m)#(M,1,K)
u_embedding = Flatten()(u_embedding)#(N,K)
m_embedding = Flatten()(m_embedding)#(M,K)
X = Concatenate()([u_embedding, m_embedding])#(N,2K)

#Dense layers of NN, the structure can be replaced and modified
#and can be added regularization
X = Dense(256)(X)
X = Activation('relu')(X)
X = Dense(100)(X)
X = Activation('relu')(X)
X = Dense(1)(X)

#Compile and train the model
model = Model(inputs=[u, m], outputs=X)
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test))

#Plot the loss function for train and validation data
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.show()

#Plot the mse for train and validation data
plt.plot(history.history['mean_squared_error'], label='mse')
plt.plot(history.history['val_mean_squared_error'], label='val mse')
plt.show()

#Save the model
model.save("recommender_model.h5")