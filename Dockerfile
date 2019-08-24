FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="writtic@gmail.com"
LABEL version="0.1"

WORKDIR /mnist-api
ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update -qq && apt-get install -yq --no-install-recommends \
    build-essential git curl wget llvm make cmake \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION
ENV PYENV_ROOT ${HOME}/.pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    apt-get remove -yq curl wget


COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PATH=$PATH:/usr/local/cuda-9.0
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64

COPY . .

ENTRYPOINT ["gunicorn", "main:app", "--bind=0.0.0.0:5000", "--log-file=-"]
