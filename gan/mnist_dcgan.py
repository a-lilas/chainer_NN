# coding:utf-8
'''
DCGANを用いたMNIST文字生成
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
from chainer.datasets import get_mnist
import util
from pycrayon import CrayonClient
import time


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__(
            # チャンネル数は元論文の半分
            fc0=L.Linear(None, 4*4*128),
            bn1=L.BatchNormalization(4*4*128),
            # 4 -> 8
            deconv2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            bn3=L.BatchNormalization(64),
            # 8 -> 14
            deconv4=L.Deconvolution2D(64, 32, 4, stride=2, pad=2),
            bn5=L.BatchNormalization(32),
            # 14 -> 26
            deconv6=L.Deconvolution2D(32, 16, 4, stride=2, pad=2),
            bn7=L.BatchNormalization(16),
            # 26 -> 28
            # Batch Normalization は用いない
            deconv8=L.Deconvolution2D(16, 1, 5, stride=1, pad=1, outsize=(28, 28))
            )

    def __call__(self, z):
        h = F.relu(self.bn1(self.fc0(z)))
        h = F.reshape(h, (z.data.shape[0], 128, 4, 4))
        h = F.relu(self.bn3(self.deconv2(h)))
        h = F.relu(self.bn5(self.deconv4(h)))
        h = F.relu(self.bn7(self.deconv6(h)))
        y = F.tanh(self.deconv8(h))

        return y

       
class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            # 逆畳み込みと畳み込みにおいて、各種パラメータを同じにすると、
            # サイズの変化は対称的なものになる。
            # 28 -> 26
            # Batch Normalization は用いない
            conv1=L.Convolution2D(1, 16, 5, stride=1, pad=1),
            # 26 -> 14
            conv2=L.Convolution2D(16, 32, 4, stride=2, pad=2),
            bn3=L.BatchNormalization(32),
            # 14 -> 8
            conv4=L.Convolution2D(32, 64, 4, stride=2, pad=2),
            bn5=L.BatchNormalization(64),
            # 8 -> 4
            conv6=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            bn7=L.BatchNormalization(128),
            # 4*4*128 -> 1 (is_gen or is_real)
            fc8=L.Linear(4*4*128, 1)
        )

    def __call__(self, x):
        # slope = 0.2 (default)
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv6(h))
#        h = F.leaky_relu(self.bn3(self.conv2(h)))
#        h = F.leaky_relu(self.bn5(self.conv4(h)))
#        h = F.leaky_relu(self.bn7(self.conv6(h)))
        y = self.fc8(h)

        return y


if __name__ == '__main__':
    # GPUフラグ
    gpu_fg = util.gpuCheck(sys.argv)
    if gpu_fg >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_fg >= 0 else np

    # pycrayon 初期化
    cc = CrayonClient(hostname="192.168.1.201", port=8889)
    # delete this experiment from the server
    try:
        cc.remove_experiment("MNIST_DCGAN_GEN")
        cc.remove_experiment("MNIST_DCGAN_DIS")
    except:
        pass

    # create a new experiment
    try:
        tb_gen = cc.create_experiment("MNIST_DCGAN_GEN")
        tb_dis = cc.create_experiment("MNIST_DCGAN_DIS")
    except:
        tb_gen = cc.open_experiment("MNIST_DCGAN_GEN")
        tb_dis = cc.open_experiment("MNIST_DCGAN_DIS")
        

    # Training Data
    train, test = chainer.datasets.get_mnist()
    x_train, t_train = train._datasets
    x_train = (x_train - 0.5) / 0.5
    x_test, t_test = test._datasets
    # 学習データ数
    train_size = len(x_train)
    # テストデータ数
    test_size = len(x_test)
    # エポック数
    epoch_n = 50
    # バッチサイズ
    batch_size = 50
    # ノイズZの次元数
    z_dim = 75

    # model
    gen = Generator()
    dis = Discriminator()

    # to GPU
    if gpu_fg >= 0:
        cuda.get_device(gpu_fg).use()
        gen.to_gpu()
        dis.to_gpu()

    # optimizer
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0001, beta1=0.2)
    o_gen.setup(gen)
    o_dis.setup(dis)
#    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
#    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    for epoch in range(epoch_n):
        start_time = time.time()
        sum_L_gen = 0
        sum_L_dis = 0
        acc_sum = 0
        perm = np.random.permutation(train_size)

        for i in range(0, train_size, batch_size):
            # print("train generator")
            # train generator
            # make noise z
            z = xp.random.uniform(-1, 1, (batch_size, z_dim), dtype=np.float32)
            z = Variable(z)
            x = gen(z)
            # 全て識別器から発生したデータで学習させる
            y = dis(x)
            # 識別器の正解
            # 0: from dataset
            # 1: from noise
            # L_gen: 識別器の結果が、全て0だとOK（識別器を騙せている）
            # L_dis: 識別器の結果が、全て1だとOK（生成器を見破れている）
            L_gen = F.sigmoid_cross_entropy(y, Variable(xp.zeros((batch_size, 1), dtype=np.int32)))
            L_dis = F.sigmoid_cross_entropy(y, Variable(xp.ones((batch_size, 1), dtype=np.int32)))

            # print("train discriminator")
            # train discriminator
            # batch data (MNIST)
            x = Variable(xp.asarray(xp.reshape(x_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]], (batch_size, 1, 28, 28))))
            t = Variable(xp.asarray(t_train[perm[i:(i+batch_size) if (i+batch_size) < train_size else train_size]]))
#            print(x.data[0], np.shape(x.data[0]), np.mean(x.data[0]), np,max(x.data[0]))
#            exit()
            # 全てデータセット中のデータで学習させる
            y = dis(x)
            # 全てデータセット中のデータなので、識別器の結果が全て0だとOK
            L_dis += F.sigmoid_cross_entropy(y, Variable(xp.zeros((batch_size, 1), dtype=np.int32)))
            
            gen.zerograds()
            L_gen.backward()
            o_gen.update()
            
            dis.zerograds()
            L_dis.backward()
            o_dis.update()

            sum_L_gen += L_gen.data * batch_size
            sum_L_dis += L_dis.data * batch_size

        if epoch % 1 == 0:
            # save the image
            z = xp.random.uniform(-1, 1, (25, z_dim), dtype=np.float32)
            z = Variable(z)
            x = gen(z)

            # need to send to CPU from GPU            
            # x = cuda.to_cpu(x.data.astype(int))
            x = cuda.to_cpu(x.data)
            x = np.reshape(x, (-1, 28, 28))
            for i in range(25):
                print(x[i, 0, 0:5], np.mean(x[i]))
                plt.subplot(5, 5, i+1)
                plt.axis('off')
                plt.imshow(x[i], cmap='gray')
            plt.savefig("epoch{}.png".format(epoch))

        print('epoch: {}'.format(epoch))
        # 訓練誤差, 正解率, 学習時間
        print('sigmoid cross entropy (gen): {}'.format(sum_L_gen / train_size))
        print('sigmoid cross entropy (dis): {}'.format(sum_L_dis / train_size))
        print('time per epoch: {} [sec]'.format(time.time() - start_time))
        print(' - - - - - - - - - ')

        tb_dis.add_scalar_value("sigmoid cross entropy", float(sum_L_dis/train_size))
        tb_gen.add_scalar_value("sigmoid cross entropy", float(sum_L_gen/train_size))

    # save the model
    gen.to_cpu()
    _pickle.dump(gen, open('generator.pkl', 'wb'), -1)
    # save the TensorBoard
    tb_dis.to_zip()
    tb_gen.to_zip()

