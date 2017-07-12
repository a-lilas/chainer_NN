# chainer_NN
Chainerを使って画像系の深層学習を実装して遊びます。

## 環境
- macOS Sierra 10.12.5 (pycrayon server)
- Ubuntu 16.04 (GPU server)
- Docker nvidia-docker を用いた環境構築
- ```crayon(pycrayon)```を用いた```TensorBoard```による簡易的な可視化（試験運用）

## 環境構築メモ
以下の手順に従って各種環境を導入します
1. Dockerのインストール  
[Docker公式サイト](https://docs.docker.com/docker-for-mac/)を見て頑張る。Macでの可視化、Ubuntuでのプログラム実行をそれぞれ行いたかったので、両方にDockerをインストール。
2. nvidia-dockerの導入  
[参考HP](http://blog.amedama.jp/entry/2017/04/03/235901)を見て頑張る。他にもいろいろ情報はあるかと思います。今回は、GPUがあるUbuntuに導入。
3. ```chainer```, ```CUDA```, ```cuDNN```のインストール  
バージョン管理や、割と面倒な```CUDA```周りのインストール作業を楽に済ませるため、各種モジュールが含まれるDockerイメージを作成して、コンテナを作る。Dockerイメージはバージョンが合ったものが見つからなかったのでざっくり作成。
4. ```crayon```, ```pycrayon```のインストール  
どちらもインストールについては、[このページ](https://github.com/torrvision/crayon)を参照してインストール。
