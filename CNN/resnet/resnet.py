# coding:utf-8
'''
ResNetを用いたCIFAR10物体認識
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


class ResBlock(Chain):
    # ResNet one block
    def __init__(self, input_size, output_size, stride=1):
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(input_size, output_size, 3, stride=stride, pad=1),
            bn2=L.BatchNormalization(output_size),
            conv3=L.Convolution2D(output_size, output_size, 3, stride=1, pad=1),
            bn4=L.BatchNormalization(output_size)
        )
    
    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = self.bn4(h)
        # xとhのサイズが違った場合の処理
        if x.data.shape != h.data.shape:
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = Variable(p)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        y = F.relu(h+x)
        return y


class Resnet(Chain):
    # Like Resnet for CIFAR-10
    # n: the number of ResBlock
    def __init__(self, block, n=18):
        super(Resnet, self).__init__(
            conv=L.Convolution2D(None, 16, 3, stride=1, pad=0),
            bn=L.BatchNormalization(16),
            fc=L.Linear(None, 10)
        )
        # None: 出力ノード数の自動推定
        # Like ResNet
        # conv1 = L.Convolution2D(None, 16, 3, stride=1, pad=0)
        # bn1 = L.BatchNormalization(16)
#        self.conv1 = L.Convolution2D(None, 16, 3, stride=1, pad=0)
#        self.bn1 = L.BatchNormalization(16)
#        self.fc1 = L.Linear(None, 10) 
        self.res = []
        for i in range(n):
            self.res.append(block(16, 16))
        for i in range(n):
            if i == 0:
                self.res.append(block(16, 32, 2))
            else:
                self.res.append(block(32, 32))
        for i in range(n):
            if i == 0:
                self.res.append(block(32, 64, 2))
            else:
                self.res.append(block(64, 64))
        # fc1 = L.Linear(None, 10)
        for f in self.res:
            f.to_gpu()

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        for f in self.res:
            h = f(h)
        y = self.fc(h)
        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = util.gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

    # pycrayon 初期化
    cc = CrayonClient(hostname="192.168.1.90", port=8889)
    # delete this experiment from the server
    try:
        cc.remove_experiment("ResNet train")
        cc.remove_experiment("ResNet test")
    except:
        pass

    # create a new experiment
    try:
        tb_res_train = cc.create_experiment("ResNet train")
        tb_res_test = cc.create_experiment("ResNet test")
    except:
        tb_res_train = cc.open_experiment("ResNet train")
        tb_res_test = cc.open_experiment("ResNet test")

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
    model = Resnet(block=ResBlock, n=18)
    # to GPU
    if gpu_fg >= 0:
        cuda.get_device(gpu_fg).use()
        model.to_gpu(gpu_fg)

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
        tb_res_train.add_scalar_value("softmax cross entropy -ResNet", float(loss_sum/train_size))
        tb_res_train.add_scalar_value("Accuracy -ResNet", float(acc_sum_train/train_size))
        tb_res_test.add_scalar_value("Accuracy -ResNet", float(acc_sum_test/test_size))

        # append list to plot
        loss_list.append(float(loss_sum/train_size))
        acc_sum_train_list.append(float(acc_sum_train/train_size))
        acc_sum_test_list.append(float(acc_sum_test/test_size))

    # save the model (dump)
    model.to_cpu()
    _pickle.dump(model, open("ResNet.pkl", "wb"), -1)

    np.save('ResNet_loss.npy', loss_list)
    np.save('ResNet_acc_train.npy', acc_sum_train_list)
    np.save('ResNet_acc_test.npy', acc_sum_test_list)

    # save the pycrayon data
    tb_res_train.to_zip()
    tb_res_test.to_zip()
