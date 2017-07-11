# coding:utf-8
'''
AlexNetを用いたCIFAR10物体認識
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
from chainer.datasets import get_cifar10
import util
from pycrayon import CrayonClient
import time


class Alexnet(Chain):
    # Like Alexnet for CIFAR-10
    def __init__(self, input_size):
        '''
        AlexNet
        conv1=L.Convolution2D(None, 96, 11, stride=1),
        conv2=L.Convolution2D(None, 256, 5, stride=1, pad=1),
        conv3=L.Convolution2D(None, 384, 3, stride=1, pad=1),
        conv4=L.Convolution2D(None, 384, 3, stride=1, pad=1),
        conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1),
        fc6=L.Linear(None, 4096),
        fc7=L.Linear(None, 4096),
        fc8=L.Linear(None, 1000)
        '''
        super(Alexnet, self).__init__(
            # None: 出力ノード数の自動推定
            # Like AlexNet
            conv1=L.Convolution2D(None, 64, 5, stride=1),
            bn1=L.BatchNormalization(64),
            conv2=L.Convolution2D(None, 128, 5, stride=1, pad=1),
            bn2=L.BatchNormalization(128),
            conv3=L.Convolution2D(None, 192, 3, stride=1, pad=1),
            conv4=L.Convolution2D(None, 192, 3, stride=1, pad=1),
            conv5=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            fc6=L.Linear(None, 1024),
            fc7=L.Linear(None, 512),
            fc8=L.Linear(None, 10)
            )

    def __call__(self, x, train=True):
        # x: 入力， train: 学習時はTrueに．テスト時はFalseにする
        # LCN: 局所コンストラスト正規化
        # ある1つのチャンネルについて画素全体で正規化する
        # LRN: ? (Alexnetはこっち)
        # ある画素(局所領域)について，多チャンネル全体で正規化する
        # BatchNormalization との違いは？
        # 本の"norm"と"pool"の順番ってこれでいいの？逆では？

        ######################################
        h = self.conv1(x)
        # h = F.local_response_normalization(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        ######################################
        h = self.conv2(h)
        # h = F.local_response_normalization(h)
        h = self.bn2(h)
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
        h = self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h)
        ######################################
        h = self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h)
        ######################################
        y = self.fc8(h)
        ######################################

        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = util.gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

    # pycrayon 初期化
    cc = CrayonClient(hostname="192.168.1.198", port=8889)
    # delete this experiment from the server
    try:
        cc.remove_experiment("AlexNet train (Adam)")
        cc.remove_experiment("AlexNet test (Adam)")
    except:
        pass

    # create a new experiment
    try:
        tb_alex_train = cc.create_experiment("AlexNet train (Adam)")
        tb_alex_test = cc.create_experiment("AlexNet test (Adam)")
    except:
        tb_alex_train = cc.open_experiment("AlexNet train (Adam)")
        tb_alex_test = cc.open_experiment("AlexNet test (Adam)")

    # x_train: 32*32*3
    train, test = get_cifar10()
    x_train, t_train = train._datasets
    x_test, t_test = test._datasets

    # 学習データサイズ
    input_size = 32
    # 学習データ数
    train_size = len(x_train)
    # テストデータ数
    test_size = len(x_test)
    # エポック数
    epoch_n = 150
    # バッチサイズ
    batch_size = 128
    # for plot
    loss_list = []
    acc_sum_test_list = []
    acc_sum_train_list = []

    # 2次元配列を4次元配列に変換(枚数とチャンネル数を追加)
    x_train = np.asarray(np.reshape(x_train, (train_size, 3, 32, 32)))
    x_test = np.asarray(np.reshape(x_test, (test_size, 3, 32, 32)))
    t_train = np.asarray(t_train)
    t_test = np.asarray(t_test)
    
    # pre-process of images
    x_train = (x_train - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5

    # model
    model = Alexnet(input_size=input_size)
    # to GPU
    if gpu_fg >= 0:
        cuda.get_device(gpu_fg).use()
        model.to_gpu()

    # optimizer
    optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD()
    optimizer.setup(model)

    # Training Loop
    for epoch in range(0, epoch_n):
        # time per epoch
        start_time = time.time()

        # 誤差 初期値
        loss_sum = 0
        # 精度
        acc_sum_train = 0
        acc_sum_test = 0
        # 学習時のバッチのシャッフル(only np)
        perm = np.asarray(np.random.permutation(train_size))

        if epoch % 50 == 0:
            optimizer.alpha = optimizer.alpha * 0.1


        # バッチ単位での学習
        for i in range(0, train_size, batch_size):
            # x: データ, t: ラベル
            # バッチ作成
            x_batch = xp.asarray(x_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]])
            t_batch = xp.asarray(t_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]])
            x_batch = x_batch.astype(xp.float32)
            # インデックスが要素数をオーバーした場合の処理
            x = Variable(x_batch)
            t = Variable(t_batch)
            # 勾配のゼロ初期化
            model.zerograds()
            # y: 予測(学習)
            y = model(x)
            # 損失関数(ソフトマックス->交差エントロピー)
            loss = F.softmax_cross_entropy(y, t)
            # 誤差逆伝搬
            loss.backward()
            loss.unchain_backward()
            # 損失関数を計算
            # 出力時は，".data"を参照
            loss_sum += loss.data * len(y)
            # 最適化
            optimizer.update()
            # 識別率を計算
            acc = F.accuracy(y, t)
            acc_sum_train += float(acc.data) * len(y)

        # バッチ単位でのテスト
        for i in range(0, test_size, batch_size):
            # x: データ, t: ラベル
            x_batch = xp.asarray(x_test[i:(i+batch_size) if (i+batch_size) < test_size else test_size])
            t_batch = xp.asarray(t_test[i:(i+batch_size) if (i+batch_size) < test_size else test_size])
            x_batch = x_batch.astype(xp.float32)
            x = Variable(x_batch)
            t = Variable(t_batch)

            # test mode
            with chainer.using_config('train', False):
                y = model(x)

                # 識別率を計算
                acc = F.accuracy(y, t)
                acc_sum_test += float(acc.data) * len(y)

        # 訓練誤差, 識別率, 学習時間
        print('epoch: {}'.format(epoch))
        print('softmax cross entropy: {}'.format(loss_sum / train_size))
        print('accuracy(train data): {}'.format(acc_sum_train / train_size))
        print('accuracy(test data): {}'.format(acc_sum_test / test_size))
        print('time per epoch: {} [sec]'.format(time.time() - start_time))
        print(' - - - - - - - - - ')

        # send to pycrayon server
        tb_alex_train.add_scalar_value("softmax cross entropy -AlexNet", float(loss_sum/train_size))
        tb_alex_train.add_scalar_value("Accuracy -AlexNet", float(acc_sum_train/train_size))
        tb_alex_test.add_scalar_value("Accuracy -AlexNet", float(acc_sum_test/test_size))

        # append list to plot
        loss_list.append(float(loss_sum/train_size))
        acc_sum_train_list.append(float(acc_sum_train/train_size))
        acc_sum_test_list.append(float(acc_sum_test/test_size))

    # save the model (dump)
    model.to_cpu()
    _pickle.dump(model, open("AlexNet.pkl", "wb"), -1)

    np.save('AlexNet_loss.npy', loss_list)
    np.save('AlexNet_acc_train.npy', acc_sum_train_list)
    np.save('AlexNet_acc_test.npy', acc_sum_test_list)

    # save the pycrayon data
    tb_alex_train.to_zip()
    tb_alex_test.to_zip()
