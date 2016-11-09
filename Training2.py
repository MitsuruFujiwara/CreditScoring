# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation

class TrainingBase(object):

    def __init__(self, datapath, ylabel, numTraining, init_w):
        self.df = pd.read_csv(datapath)
        self.Y = df[ylabel] # Y value
        self.X = df.drop(ylabel, 1) # X value
        self.numTraining = numTraining # number of training
        self.numClass = None # number of classes
        self.w = init_w # initial weights

    def vecorConverter(self, y):
        # convert target value into vector
        for i, t in enumerate(y):
            yield np.eye(1, self.numClass)

    def model(self):
        _model = Sequential()
        _model.add(Dense(output_dim=238, input_dim=trX.shape[1]))
        _model.add(Activation('tanh'))
        _model.add(Dense(output_dim=238, input_dim=238))
        _model.add(Activation('tanh'))
        _model.add(Dense(output_dim=numClass, input_dim=238))
        _model.add(Activation('softmax'))
        return _model

    def saveResult(self, outputpath):
        

    def fit(self):
        # set data
        trX = np.array(self.X)
        trY = np.array(list(self.vecorConverter(self.Y)))

        # set model
        model = self.model()

        # set itinial weights
        model.set_weights(self.w)

        # compile
        model.compile()

        # training
        result = model.fit(self.)

if __name__ == '__main__':
