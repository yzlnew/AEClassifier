#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def split(filename):
    """split data file

    :filename: TODO
    :returns: TODO

    """
    data = np.loadtxt(open(filename, "rb"), dtype=str,
                      usecols=0, delimiter=",", skiprows=0)
    pos_tuple = np.where(data == 'AMP_V')
    pos = pos_tuple[0]
    print(len(pos))
    j = 0

    for i in range(len(pos) - 1):
        np.savetxt('yl' + str(j) + '.csv',
                   data[(pos[i] - 3):(pos[i + 1] - 4)],
                   fmt='%s', delimiter=",")
        print('generated')
        j = j + 1


if __name__ == "__main__":
    split('yl_raw.csv')
