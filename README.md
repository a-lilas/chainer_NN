# chainer_NN

## 環境構築
* Python 3.6.0
* Anaconda 4.3.0
    * numpy
    * scipy
    * matplotlib
* opencv3  
    以下のコマンドを実行してインストール  
    ```<envname>``` にはインストールする仮想環境名を入れる  
    ```
    conda install -n <envname> --channel https://conda.anaconda.org/menpo opencv3
    ```
    以下のコマンドを実行して正しくインストールできたかどうか確認  
    ```python
    import cv2
    cv2.__version__
    ```
* chainer  
    仮想環境内で以下のコマンドを実行してインストール
    ```
    pip install chainer
    ```