# coding:utf-8
'''
CNNを用いたMNIST手書き文字識別
'''
import numpy as np
import sys
import matplotlib.pyplot as plt
import _pickle
import chainer
from chainer import cuda
from chainer import Function, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def gpuCheck(argv):
    # GPU フラグのチェック
    if len(argv) != 1:
        print('Error. "python mnist_cnn.py [-cpu] or [-gpu]')
        exit()
    if argv[1] == '-gpu':
        return 0
    elif argv[1] == '-cpu':
        return -1
    else:
        print('Error. "python mnist_cnn.py [-cpu] or [-gpu]')
        exit()


class CNN(Chain):
    # CNN for MNIST
    # http://qiita.com/To_Murakami/items/35d1b3144a0d017ad0ee
    # http://qiita.com/qooa/items/b671e12ac8302fe977d3
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            l3=L.Linear(7*7*64, 1024),
            l4=L.Linear(1024, 10)
            )

    def __call__(self, x, train=True):
        # x: 入力， train: 学習時はTrueに．テスト時はFalseにする

        # 2*2でmaxpooling
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l3(h))
        # ドロップアウト, ratio: 割合,train: 学習時のみドロップアウトする
        # 引数trainはver.2以降，サポートされなくなった
        h = F.dropout(h, ratio=0.5)
        y = self.l4(h)

        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

    # Training Data
    train, test = chainer.datasets.get_mnist()
    x_train, t_train = train._datasets
    x_test, t_test = test._datasets

    # 学習データ数
    train_size = len(x_train)
    # 教師データ数
    test_size = len(x_test)
    # エポック数
    epoch_n = 15
    # バッチサイズ
    batch_size = 100

    # 2次元配列を4次元配列に変換(枚数とチャンネル数を追加)
    x_train = xp.asarray(xp.reshape(x_train, (train_size, 1, 28, 28)))
    x_train = x_train.astype(xp.float32)
    x_test = xp.asarray(xp.reshape(x_test, (test_size, 1, 28, 28)))
    x_test = x_test.astype(xp.float32)

    # model
    model = CNN()
    # to GPU
    if gpu_fg >= 0:
        cuda.get_device(gpu_fg).use()
        model.to_gpu()

    # optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Training Loop
    for epoch in range(0, epoch_n):
        # 誤差 初期値
        loss_sum = 0
        # 精度
        acc_sum = 0
        # バッチのシャッフル
        perm = xp.random.permutation(train_size)

        # バッチ単位での学習
        for i in range(0, train_size, batch_size):
            # x: データ, t: 教師
            # バッチ作成
            # インデックスが要素数をオーバーした場合の処理
            x = Variable(x_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]])
            t = Variable(t_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]])
            # 勾配のゼロ初期化
            model.zerograds()
            # y: 予測(学習)
            y = model(x)
            # 損失関数(ソフトマックス->交差エントロピー)
            loss = F.softmax_cross_entropy(y, t)
            # 誤差逆伝搬
            loss.backward()
            # 誤差と正解率を計算
            # 出力時は，".data"を参照
            loss_sum += loss.data * batch_size
            # 最適化
            optimizer.update()

        # バッチ単位でのテスト
        # エポックごとにテストデータを用いて評価を行う(もちろん誤差逆伝播は行わない)
        for i in range(0, test_size, batch_size):
            # x: データ, t: 教師
            # バッチ作成
            # インデックスが要素数をオーバーした場合の処理
            x = Variable(x_test[i:(i+batch_size) if (i+batch_size) < test_size else test_size])
            t = Variable(t_test[i:(i+batch_size) if (i+batch_size) < test_size else test_size])
            # y: 予測(学習)
            y = model(x)
            # 精度を計算
            acc = F.accuracy(y, t)
            acc_sum += float(acc.data) * len(y)

        print('epoch: {}'.format(epoch))
        # 訓練誤差, 正解率
        print('softmax cross entropy = {}'.format(loss_sum / train_size))
        print('accuracy: {}'.format(acc_sum / test_size))
        print(' - - - - - - - - - ')

    # save the model (dump)
    model.to_cpu()
    _pickle.dump(model, open("model.pkl", "wb"), -1)