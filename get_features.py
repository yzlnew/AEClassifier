#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import speechpy as spy

from load_data import load_data


def get_mfcc(filename):
    """get mfcc features

    :filename: TODO
    :returns: TODO

    """
    sampling_frez = 10000
    data = load_data(filename)
    data_norm = data / max(data)
    features = spy.feature.mfcc(data_norm, sampling_frez)
    features_avg = np.sum(features, axis=0)
    # print(features_avg.size)

    return features_avg


if __name__ == "__main__":
    dq_feature = np.array([])
    yl_feature = np.array([])
    file_sum = 100
    feature_sum = 13
    train_sum = int(file_sum * 0.8)
    validate_sum = file_sum - train_sum

    for i in range(file_sum):
        filename = 'dq/dq' + str(i) + '.csv'
        feature = get_mfcc(filename)
        dq_feature = np.append(dq_feature, feature)
        filename = 'yl/yl' + str(i) + '.csv'
        feature = get_mfcc(filename)
        yl_feature = np.append(yl_feature, feature)

    dq_feature = np.reshape(dq_feature, (file_sum, feature_sum))
    yl_feature = np.reshape(yl_feature, (file_sum, feature_sum))
    # print(dq_feature)
    np.save('dq_train', dq_feature[0:train_sum, ...])
    np.save('yl_train', yl_feature[0:train_sum, ...])
    # reload = np.load('dq_train' + '.npy')
    # print(reload.size)
    np.save('dq_validate', dq_feature[train_sum:, ...])
    np.save('yl_validate', yl_feature[train_sum:, ...])


