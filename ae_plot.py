#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import speechpy as spy


def load_data(filename):
    """load data from file

    :filename: TODO
    :returns: TODO

    """
    return np.loadtxt(open(filename, "rb"),
                      usecols=0, delimiter=",", skiprows=4)


def ae_plot(filename):
    """Plot AE signal

    :filename: TODO
    :returns: TODO

    """
    data = load_data(filename)
    data_norm = data/max(data)
    plt.figure()
    plt.plot(data)
    plt.savefig(filename+'_fig'+'.png')
    transformed = np.fft.fft(data_norm)
    trans_abs = abs(transformed)
    plt.figure()
    plt.plot(data_norm)
    plt.savefig(filename+'_norm'+'.png')
    plt.figure()
    plt.plot(np.log10(trans_abs))
    plt.savefig(filename+'_fft'+'.png')


def mfcc_plot(filename):
    """TODO: Docstring for mfcc_plot.

    :filename: TODO
    :returns: TODO

    """
    sampling_frez = 10000
    data = load_data(filename)
    data_norm = data/max(data)
    results = spy.feature.mfcc(data_norm, sampling_frez)
    plt.figure()
    plt.plot(results)
    plt.savefig(filename+'_mfcc'+'.png')


def mfe_plot(filename):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    sampling_frez = 10000
    data = load_data(filename)
    data_norm = data/max(data)
    [features, results] = spy.feature.mfe(data_norm, sampling_frez)
    plt.figure()
    plt.plot(results)
    plt.savefig(filename+'_mfe'+'.png')


if __name__ == '__main__':
    for i in range(3):
        ae_plot('yl/yl'+str(i)+'.csv')
        ae_plot('dq/dq'+str(i)+'.csv')
