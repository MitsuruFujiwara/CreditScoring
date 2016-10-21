# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation

# load dataset
df = pd.read_csv('TrainingData.csv')
ratings = pd.read_csv('RatingsTable.csv')

# number of training
numTraining = 80000

# set paramters
classlist = list(ratings['#'])
numClass = len(classlist)

# set data
Y = df['Ratings #']
X = df.drop('Ratings #', 1)

# convert data into vector
def __trY(y):
    for i, t in enumerate(y):
        yield np.eye(1, numClass, t-1)

# set data for training
trY = np.array(list(__trY(Y))).reshape(len(Y), numClass)
trX = np.array(X)

# set model
model = Sequential()
model.add(Dense(output_dim=300, input_dim=trX.shape[1]))
model.add(Activation('relu'))
model.add(Dense(output_dim=300, input_dim=300))
model.add(Activation('relu'))
model.add(Dense(output_dim=numClass, input_dim=300))
model.add(Activation('softmax'))

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
json_string = model.to_json()
open('model.json', 'w').write(json_string)

# save parameters
model.save_weights('param.h5')
