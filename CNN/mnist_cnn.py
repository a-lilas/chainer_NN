# coding:utf-8
'''
CNNを用いたMNIST手書き文字識別
'''
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import cuda
from chainer import Function, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


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
    # Training Data
    train, test = chainer.datasets.get_mnist()
    x_train, t_train = train._datasets
    x_test, t_test = test._datasets

    # 学習データ数
    train_size = len(x_train)
    # 教師データ数
    test_size = len(x_test)
    # エポック数
    epoch_n = 20
    # バッチサイズ
    batch_size = 100

    # 2次元配列を4次元配列に変換(枚数とチャンネル数を追加)
    x_train = np.asarray(np.reshape(x_train, (train_size, 1, 28, 28)))
    x_train = x_train.astype(np.float32)
    x_test = np.asarray(np.reshape(x_test, (test_size, 1, 28, 28)))
    x_test = x_test.astype(np.float32)

    # model ,optimizer
    model = CNN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Training Loop
    for epoch in range(0, epoch_n):
        # 誤差 初期値
        loss_val = 0
        # バッチのシャッフル
        perm = np.random.permutation(train_size)

        for i in range(0, train_size, batch_size):
            # x: データ, t: 教師
            # バッチ作成
            if (i+batch_size) < train_size:
                x = Variable(x_train[perm[i:(i+batch_size)]])
                t = Variable(t_train[perm[i:(i+batch_size)]])
            else:
                # インデックスが要素数をオーバーした場合の処理
                x = Variable(x_train[perm[i:train_size]])
                t = Variable(t_train[perm[i:train_size]])

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
            loss_val += loss.data * batch_size
            # 最適化
            optimizer.update()

        print('epoch:', epoch)
        # 訓練誤差, 正解率
        print('softmax cross entropy = {}'.format(loss_val))
        print(' - - - - - - - - - ')

    # Test Loop
    # x: データ, t: 教師
    # バッチ作成
    x = Variable(x_test)
    t = Variable(t_test)
    # 勾配のゼロ初期化
    model.zerograds()
    # y: 予測(学習)
    y = model(x)
    # 損失関数(ソフトマックス->交差エントロピー)
    loss = F.softmax_cross_entropy(y, t)
    # 精度
    accuracy = F.accuracy(y, t)
    # 誤差逆伝搬
    loss.backward()
    # 誤差と正解率を計算
    # 出力時は，".data"を参照
    loss_val += loss.data * batch_size
    # 最適化
    optimizer.update()

    print('--- Test ---')
    print('accuracy: %f' % accuracy)
    print('loss_val: %f' % loss_val)
