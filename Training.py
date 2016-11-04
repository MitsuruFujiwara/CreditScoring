# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD

# load dataset
df = pd.read_csv('TrainingData_rev.csv')
ratings = pd.read_csv('RatingsTable.csv')

# number of training
numTraining = 10000

# set paramters
classlist = list(ratings['#'])
numClass = len(classlist) # 22

# set data
Y = df['Ratings #']
X = df.drop('Ratings #', 1).fillna(0)

# convert data into vector
def __trY(y):
    for i, t in enumerate(y):
        yield np.eye(1, numClass, t-1)

# set data for training
trY = np.array(list(__trY(Y))).reshape(len(Y), numClass)
trX = np.array(X)

# set model
model = Sequential()
model.add(Dense(output_dim=238, input_dim=trX.shape[1]))
model.add(Activation('tanh'))
model.add(Dense(output_dim=238, input_dim=238))
model.add(Activation('tanh'))
model.add(Dense(output_dim=numClass, input_dim=238))
model.add(Activation('softmax'))

# load autoencoder
encoder1 = load_model('encoder1.h5')
encoder2 = load_model('encoder2.h5')
encoder3 = load_model('encoder3.h5')

# set initial weights
w = model.get_weights()
w[0] = encoder1.get_weights()[0]
w[1] = encoder1.get_weights()[1]
w[2] = encoder2.get_weights()[0]
w[3] = encoder2.get_weights()[1]
w[4] = encoder3.get_weights()[0]
w[5] = encoder3.get_weights()[1]
model.set_weights(w)

# compile
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# training
his = model.fit(trX, trY, nb_epoch=numTraining, verbose=2)

# plot result
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(his.history['loss'])

# save figure
plt.savefig('loss.png')

# save model
model.save('model.h5')
