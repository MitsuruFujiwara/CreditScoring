# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras.models import Model

class AutoEncoderBase(object):
    """
    Base class of AutoEncoder

    input:
        datapath: datapath (csv)
        ylabel: label for data Y (string)
        input_dim: input dimension (integer)
        output_dim: output dimension (integer)
        numLayer: number of layer (integer, default: 3)
        numTraining: number of training (long, default: 5000)
        activation: activation function (default: relu)

    output:
        result: list of encoder object

    """

    def __init__(self, datapath, ylabel, input_dim, output_dim, numLayer=3, numTraining=5000, activation='relu'):
        self.trX = pd.read_csv(datapath).drop(ylabel, 1).fillna(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = round((self.input_dim+self.output_dim)/2)
        self.numLayer = numLayer
        self.numTraining = numTraining
        self.activation = activation
        self.result = []

    def encoding_dim(self):
        # set encoding dim
        for i in range(self.numLayer):
            if i == self.numLayer - 1:
                yield self.output_dim
            else:
                yield self.hidden_dim

    def input_data(self):
        # set input data
        for i in range(self.numLayer):
            if i == 0:
                yield Input(shape=(self.trX.shape[1],))
            else:
                yield Input(shape=(self.hidden_dim,))

    def encoded(self, encoding_dim, input_data):
        # set encoded layer
        for (t, u) in zip(encoding_dim, input_data):
            yield Dense(output_dim=t, activation=self.activation)(u)

    def decoded(self, encoded):
        # set decoded layer
        for i, t in enumerate(encoded):
            if i == 0:
                yield Dense(output_dim=self.input_dim, activation=self.activation)(t)
            else:
                yield Dense(output_dim=self.hidden_dim, activation=self.activation)(t)

    def autoencoder(self, input_data, decoded):
        # set autoencoder
        for (t, u) in zip(input_data, decoded):
            yield Model(input=t, output=u)

    def encoder(self, input_data, encoded):
        # set encoder for output
        for (t, u) in zip(input_data, encoded):
            yield Model(input=t, output=u)

    def fit(self):
        encoding_dim = list(self.encoding_dim())
        input_data = list(self.input_data())
        encoded = list(self.encoded(encoding_dim, input_data))
        decoded = list(self.decoded(encoded))
        autoencoder = list(self.autoencoder(input_data, decoded))
        encoder = list(self.encoder(input_data, encoded))
        _trX = []

        for i, t in enumerate(autoencoder):
            # set training data
            if i == 0:
                _trX.append(np.array(self.trX))
            else:
                _trX.append(encoder[i-1].predict(_trX[i-1]))

            X = _trX[i]

            # compile
            t.compile(loss='mse', optimizer='adagrad')

            # training
            t.fit(X, X, nb_epoch=self.numTraining, verbose=2)

            # save encoder
            self.result.append(encoder[i])

        return self.result

if __name__ == '__main__':
    # test
    datapath = 'TrainingData_rev.csv'
    ylabel = 'Ratings #'
    input_dim = 455
    output_dim = 22

    ae = AutoEncoderBase(datapath, ylabel, input_dim, output_dim, numTraining=50000, activation='tanh')
    result = ae.fit()

    # save result
    outputpath = ['encoder1.h5', 'encoder2.h5', 'encoder3.h5']
    for (u, t) in zip(result, outputpath):
        u.save(t)
