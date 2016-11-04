# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras.models import Model

class AutoEncoder(object):
    """
    Class of AutoEncoder
    input: training dataset trX, encoding_dim
    output: estimated parameters W
    """

    def __init__(self, datapath, outputpath, encoding_dim, numEpoc):
        self.encoding_dim = encoding_dim
        self.df = pd.read_csv(datapath)
        self.X = self.df.drop('Ratings #', 1).fillna(0)
        self.outputpath = outputpath
        self.numEpoc = numEpoc

    def setAutoEncoder(self):
        input_dim = len(self.X.columns)
        input_data = Input(shape=(input_dim,))

        encoded = Dense(encoding_dim, activation='relu')(input_data)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input=input_data, output=decoded)
        self.autoencoder.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    def main(self):
        trX = self.X.values
        self.setAutoEncoder()
        self.autoencoder.fit(trX, trX, nb_epoch=numEpoc, verbose=2)
        self.autoencoder.save_weights(self.outputpath)
        w = self.autoencoder.get_weights()
        print np.array(w).shape
        return w

if __name__ == '__main__':
    datapath = 'TrainingData.csv'
    outputpath = 'param_enc.h5'
    encoding_dim = 300
    numEpoc = 100

    ae = AutoEncoder(datapath, outputpath, encoding_dim, numEpoc)
    res = ae.main()
    df = pd.read_csv('TrainingData.csv')
    print df
