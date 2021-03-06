# coding:utf-8
'''
多層パーセプトロンを用いた多項式フィッティング
参考: http://btgr.hateblo.jp/entry/2016/05/21/150539 (古いかも?)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
from chainer import cuda
from chainer import Function, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# 中間層のユニット数
units_cnt1 = 3
units_cnt2 = 5


class MLP(Chain):
    # 多層パーセプトロンモデル
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(1, units_cnt1),
            l2=L.Linear(units_cnt1, units_cnt2),
            l3=L.Linear(units_cnt2, 1)
            )

    def __call__(self, x):
        # x: 入力
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        y = self.l3(h2)

        return y


def makeData(n):
    # データ生成
    # 任意の関数に対して，ガウス分布に従うノイズを付与する
    noize = np.random.normal(loc=0, scale=np.sqrt(0.15), size=n)
    noize = np.reshape(noize, (-1, 1))
    x = np.random.uniform(low=0, high=10, size=n)
    # 2次元ベクトルの形で与える?
    x = np.reshape(x, (-1, 1))
    # 任意の関数(教師データ)
    y = np.sin(2*x) + np.cos(x) + noize
    # 学習データプロット
    ax1.scatter(x, y, color='green')

    # 正解データ
    x_ans = np.arange(0, 10, 0.01)
    y_ans = np.sin(2*x_ans) + np.cos(x_ans)

    ax1.plot(x_ans, y_ans, color='blue')
    ax1.set_xlim((0, 10))
    ax1.set_ylim((-2, 2))

    return x, y


if __name__ == '__main__':
    # For visualize
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ims = []
    loss_plt = []
    epoch_plt = []

    # Training Data
    n = 100
    data_x, data_t = makeData(n=n)

    model = MLP()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Training Loop
    # 誤差
    loss_val = 100
    # エポック数
    epoch = 0

    # Test Data
    x_test = np.arange(0, 10, 0.01)
    x_test = np.reshape(x_test, (-1, 1))
    x_test = chainer.Variable(np.asarray(x_test).astype(np.float32))

    while loss_val > 0.05:
        # x: データ, t: 教師
        # 型はfloat32を推奨
        x = chainer.Variable(np.asarray(data_x).astype(np.float32))
        t = chainer.Variable(np.asarray(data_t).astype(np.float32))

        # 勾配のゼロ初期化
        model.zerograds()

        # y: 予測(学習)
        y = model(x)

        # 損失関数(平均二乗誤差)
        loss = F.mean_squared_error(y, t)
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

            # 現在のエポック数におけるテストプロット
            y_test = model(x_test)

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
