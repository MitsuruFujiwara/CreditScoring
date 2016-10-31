# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation

# load dataset
df = pd.read_csv('testData_PCA.csv')
ratings = pd.read_csv('RatingsTable.csv')

# set paramters
classlist = list(ratings['#'])
numClass = len(classlist)

# set data
Y = df['Y']
X = df.drop('Y', 1).fillna(0)

# convert data into vector
def __trY(y):
    for i, t in enumerate(y):
        yield np.eye(1, numClass, t-1)

# set data for training
trY = np.array(list(__trY(Y))).reshape(len(Y), numClass)
trX = np.array(X)

# load model
model = model_from_json(open('model_PCA.json').read())

# load parameters
model.load_weights('param_PCA.h5')

# compile
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# show results
score = model.evaluate(trX, trY, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

result = {}

result['Act'] = Y-1
result['Predict'] = model.predict_classes(trX)

result = pd.DataFrame(result)
print result
