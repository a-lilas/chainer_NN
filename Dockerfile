FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --upgrade pip
RUN apt-get update -y
RUN apt-get install -y python3-setuptools
RUN pip install chainer==2.0.0
RUN pip install cupy
RUN apt-get install git
RUN apt-get install vim