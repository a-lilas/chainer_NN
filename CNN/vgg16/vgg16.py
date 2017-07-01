# coding:utf-8
'''
VGG16を用いたCIFAR10物体認識
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


class BatchConv(Chain):
    # Convolution2D + BatchNormalization
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        super(BatchConv, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, ksize, stride, pad),
            bn=L.BatchNormalization(out_ch)
        )

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return x


class MyVGG16(Chain):
    # Like VGG16 for CIFAR-10 (+Batch Normalization)
    # http://qiita.com/dsanno/items/ad84f078520f9c9c3ed1
    def __init__(self, input_size):
        # CIFAR-10に対してはノード数などを再考する必要あり
        super(MyVGG16, self).__init__(
            bnconv1_1=BatchConv(None, 64, 3, stride=1, pad=1),
            bnconv1_2=BatchConv(None, 64, 3, stride=1, pad=1),
            bnconv2_1=BatchConv(None, 128, 3, stride=1, pad=1),
            bnconv2_2=BatchConv(None, 128, 3, stride=1, pad=1),
            bnconv3_1=BatchConv(None, 256, 3, stride=1, pad=1),
            bnconv3_2=BatchConv(None, 256, 3, stride=1, pad=1),
            bnconv3_3=BatchConv(None, 512, 3, stride=1, pad=1),
            bnconv3_4=BatchConv(None, 512, 3, stride=1, pad=1),
            fc4=L.Linear(None, 4096),
            fc5=L.Linear(None, 1024),
            fc6=L.Linear(None, 10)
            )

    def __call__(self, x, train=True):
        ######################################
        h = self.bnconv1_1(x)
        h = F.relu(h)
        h = self.bnconv1_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        # h = F.dropout(h, 0.25)
        ######################################
        h = self.bnconv2_1(h)
        h = F.relu(h)
        h = self.bnconv2_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        # h = F.dropout(h, 0.25)
        ######################################
        h = self.bnconv3_1(h)
        h = F.relu(h)
        h = self.bnconv3_2(h)
        h = F.relu(h)
        h = self.bnconv3_3(h)
        h = F.relu(h)
        h = self.bnconv3_4(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h)
        ######################################
        h = self.fc4(h)
        h = F.relu(h)
        h = F.dropout(h)
        ######################################
        h = self.fc5(h)
        h = F.relu(h)
        # h = F.dropout(h, 0.25)
        ######################################
        y = self.fc6(h)
        ######################################

        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = util.gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

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
    epoch_n = 100
    # バッチサイズ
    batch_size = 64
    # for plot
    loss_list = []
    acc_sum_test_list = []
    acc_sum_train_list = []

    # 2次元配列を4次元配列に変換(枚数とチャンネル数を追加)
    x_train = np.asarray(np.reshape(x_train, (train_size, 3, 32, 32)))
    x_test = np.asarray(np.reshape(x_test, (test_size, 3, 32, 32)))
    t_train = np.asarray(t_train)
    t_test = np.asarray(t_test)

    # model
    model = MyVGG16(input_size=input_size)
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
        acc_sum_train = 0
        acc_sum_test = 0
        # 学習時のバッチのシャッフル(only np)
        perm = np.asarray(np.random.permutation(train_size))

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

        # append list to plot
        loss_list.append(float(loss_sum/train_size))
        acc_sum_train_list.append(float(acc_sum_train/train_size))
        acc_sum_test_list.append(float(acc_sum_test/test_size))

    # save the model (dump)
    model.to_cpu()
    _pickle.dump(model, open("MyVGG16.pkl", "wb"), -1)

    # plot loss/acc
    x = np.arange(0, epoch_n, 1)
    plt.plot(x, loss_list)
    plt.title('Softmax cross entropy')
    xlabels = [0, 20, 40, 60, 80, 100]
    plt.xticks(xlabels, xlabels)
    plt.show()
    plt.plot(x, acc_sum_train_list, color='red', label='train accuracy')
    plt.plot(x, acc_sum_test_list, color='blue', label='test accuracy')
    plt.title('Accuracy')
    plt.legend(loc='lower right')
    plt.xticks(xlabels, xlabels)
    plt.show()

    np.save('MyVGG16_loss.npy', loss_list)
    np.save('MyVGG16_acc_train.npy', acc_sum_train_list)
    np.save('MyVGG16_acc_test.npy', acc_sum_test_list)
