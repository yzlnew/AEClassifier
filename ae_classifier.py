#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from time import time

def read_data(filename):
    data = np.load(filename)
    res = np.reshape(data, (-1, 13))

    return res


def main():
    dq_train = read_data('dq_train.npy')
    yl_train = read_data('yl_train.npy')
    train_data = np.concatenate((dq_train, yl_train), axis=0)
    dq_validate = read_data('dq_validate.npy')
    yl_validate = read_data('yl_validate.npy')
    validate_data = np.concatenate((dq_validate, yl_validate), axis=0)

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    validate_data = scaler.transform(validate_data)

    train_label = np.linspace(1, 159, 160) // 80
    # print(train_label.size)
    clf = MLPClassifier(hidden_layer_sizes=(15), activation='tanh',
                        max_iter=20000)
    start = time()
    clf.fit(train_data, train_label)
    end = time()
    print('training time: %3f'%(end-start))
    predictions =  clf.predict(validate_data)
    print(predictions)


if __name__ == '__main__':
    main()
