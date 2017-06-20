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
        super(MLP, self).__init__(
            conv1=F.Convolution2D(1, 32, 5, stride=1, pad=2),
            conv2=F.Convolution2D(32, 64, 5, stride=1, pad=2),
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
        h = F.dropout(h, ratio=0.5, train=train)
        y = F.self.l4(h)

        return y


if __name__ == '__main__':
    # For visualize
    fig = plt.figure()
    ims = []
    loss_plt = []
    epoch_plt = []

    # Training Data
    # 多分最初からVariableになってる
    train, test = chainer.datasets.get_mnist()
    x_train, t_train = train._datasets
    x_test, t_test = test._datasets

    # model ,optimizer
    model = CNN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Training Loop
    # 誤差 初期値
    loss_val = 100
    # エポック数
    epoch_n = 20

    for epoch in range(epoch_n):
        # x: データ, t: 教師
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
        # 最適化
        optimizer.update()

        if epoch % 1000 == 0:
            # 誤差と正解率を計算
            # 出力時は，".data"を参照
            # エポック数
            loss_val = loss.data
            print('epoch:', epoch)
            # 訓練誤差, 正解率
            print('train mean loss = {}'.format(loss_val))
            print(' - - - - - - - - - ')

            # For visualization
            loss_plt.append(loss_val)
            epoch_plt.append(epoch)
            # 複数グラフのアニメーションを行う場合は，imリストにプロットオブジェクトを格納する
            im = ax1.plot(x_test.data, y_test.data, color='red')
            # リストへの追加
            im += ax2.plot(epoch_plt, loss_plt, color='blue')
            ims.append(im)

        # n_epoch以上になると終了
        if epoch >= 25000:
            break
        epoch += 1

    ani = animation.ArtistAnimation(fig, ims, interval=300)
    ani.save('perceptron.mp4', writer='ffmpeg')
    plt.show()
