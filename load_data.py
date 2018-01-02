#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def load_data(filename):
    """load data from file

    :filename: TODO
    :returns: TODO

    """
    return np.loadtxt(open(filename, "rb"),
                      usecols=0, delimiter=",", skiprows=4)
