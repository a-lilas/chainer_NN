# coding:utf-8
'''
AlexNetを用いたMNIST手書き文字識別
'''
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


class Alexnet(Chain):
    # Alexnet for MNIST
    def __init__(self, input_size):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4)
            conv2=L.Convolution2D(96, 256, 5, stride=1, pad=1)
            conv3=L.Convolution2D(256, 384, 3, stride=1, pad=1)
            conv4=L.Convolution2D(384, 384, 3, stride=1, pad=1)
            conv5=L.Convolution2D(384, 256, 3, stride=1, pad=1)
            fc6=L.Linear(6*6*256, 4096)
            fc7=L.Linear(4096, 4096)
            fc8=L.Linear(4096, 1000)
            )

    def __call__(self, x, train=True):
        # x: 入力， train: 学習時はTrueに．テスト時はFalseにする
        # LCN: 局所コンストラスト正規化
        # ある1つのチャンネルについて画素全体で正規化する
        # LRN: ? (Alexnetはこっち)
        # ある画素(局所領域)について，多チャンネル全体で正規化する
        # 本の"norm"と"pool"の順番ってこれでいいの？逆では？

        ######################################
        h = self.conv1(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        ######################################
        h = self.conv2(x)
        h = F.local_response_normalization(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        ######################################
        h = self.conv3(h)
        h = F.relu(h)
        ######################################
        h = self.conv4(h)
        h = F.relu(h)
        ######################################
        h = self.conv5(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        ######################################
        h = F.self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h)
        ######################################
        h = F.self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h)
        ######################################
        y = F.self.fc8(h)
        ######################################

        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

    # 学習データサイズ
    input_size = 227
    # 学習データ数
    train_size = len(x_train)
    # テストデータ数
    test_size = len(x_test)
    # エポック数
    epoch_n = 20
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
        # time per epoch
        start_time = time.time()

        # 誤差 初期値
        loss_sum = 0
        # 精度
        acc_sum = 0
        # 学習時のバッチのシャッフル
        perm = np.random.permutation(train_size)

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
            # 損失関数を計算
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
        # 訓練誤差, 正解率, 学習時間
        print('softmax cross entropy: {}'.format(loss_sum / train_size))
        print('accuracy: {}'.format(acc_sum / test_size))
        print('time per epoch: {} [sec]'.format(time.time() - start_time))
        print(' - - - - - - - - - ')

    # save the model (dump)
    model.to_cpu()
    _pickle.dump(model, open("model.pkl", "wb"), -1)