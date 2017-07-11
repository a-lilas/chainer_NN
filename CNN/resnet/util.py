# coding:utf-8
import numpy as np
import sys
import matplotlib.pyplot as plt
import _pickle
import time
import chainer
from chainer import cuda
from chainer import Function, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def gpuCheck(argv):
    # GPU フラグのチェック
    if len(argv) != 2:
        print('Error. "python mnist_cnn.py [-cpu] or [-gpu]')
        exit()
    if argv[1] == '-gpu':
        return 0
    elif argv[1] == '-cpu':
        return -1
    else:
        print('Error. "python mnist_cnn.py [-cpu] or [-gpu]')
        exit()


def openPickle(filename):
    # open pickle
    # 各要素はbyte型になるため、keyを呼ぶ際はbをつける (data[b'labels']など)
    with open(filename, 'rb') as f:
        d = _pickle.load(f, encoding='bytes')
    data = d[b'data']
    labels = d[b'labels']
    data_num = len(data)

    return data, labels, data_num
