# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense
from keras.models import Model

# load dataset
df = pd.read_csv('TrainingData_rev.csv')

# number of training
numTraining = 10000

# set data
X = df.drop('Ratings #', 1).fillna(0)

# set data for training
trX = np.array(X)

"""
First layer
"""

# set parameters
endoding_dim1 = 238
input_data1 = Input(shape=(trX.shape[1],))

# set layer
encoded1 = Dense(endoding_dim1, activation='tanh')(input_data1)
decoded1 = Dense(trX.shape[1], activation='tanh')(encoded1)

# set model
autoencoder1 = Model(input=input_data1, output=decoded1)
encoder1 = Model(input=input_data1, output=encoded1)

# compile
autoencoder1.compile(loss='mse', optimizer='adagrad')

# training
autoencoder1.fit(trX, trX, nb_epoch=numTraining, verbose=2)

# get & save weights
w1 = encoder1.get_weights()
encoder1.save('encoder1.h5')

"""
Second layer
"""

# set Data
trX2 = encoder1.predict(trX)

# set parameters
endoding_dim2 = 238
input_data2 = Input(shape=(238,))

# set layer
encoded2 = Dense(endoding_dim2, activation='tanh')(input_data2)
decoded2 = Dense(238, activation='tanh')(encoded2)

# set model
autoencoder2 = Model(input=input_data2, output=decoded2)
encoder2 = Model(input=input_data2, output=encoded2)

# compile
autoencoder2.compile(loss='mse', optimizer='adagrad')

# training
autoencoder2.fit(trX2, trX2, nb_epoch=numTraining, verbose=2)

# save estimated weights
w2 = encoder2.get_weights()
encoder2.save('encoder2.h5')

"""
Third layer
"""

# set Data
trX3 = encoder2.predict(trX2)

# set parameters
endoding_dim3 = 22
input_data3 = Input(shape=(238,))

# set layer
encoded3 = Dense(endoding_dim3, activation='tanh')(input_data3)
decoded3 = Dense(238, activation='tanh')(encoded3)

# set model
autoencoder3 = Model(input=input_data3, output=decoded3)
encoder3 = Model(input=input_data3, output=encoded3)

# compile
autoencoder3.compile(loss='mse', optimizer='adagrad')

# training
autoencoder3.fit(trX3, trX3, nb_epoch=numTraining, verbose=2)

# save estimated weights
w3 = encoder3.get_weights()
encoder3.save('encoder3.h5')

# check dim
for i, t in enumerate(w1):
    print t.shape

for i, t in enumerate(w2):
    print t.shape

for i, t in enumerate(w3):
    print t.shape
