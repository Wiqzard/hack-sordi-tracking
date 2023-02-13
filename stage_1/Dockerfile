#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS build
#ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential ca-certificates ffmpeg libgl1 libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#RUN apt-get update  -y --fix-missing && \
#    apt-get install -y --no-install-recommends \
#    software-properties-common \
#    wget \
#    curl \
#    unrar \
#    unzip \
#    git && \
#    build-essential ca-certificates && \
#    apt-get clean -y && \
#    rm -rf /var/lib/apt/lists/*
#apt-get upgrade -y libstdc++6 && \
#RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
#    apt-get update && \
#    apt-get install -y gcc-9 && \
#    apt-get upgrade -y libstdc++6
# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
#    rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup


ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda

RUN conda install -n base -c conda-forge mamba

ADD ./environment.yml ./environment.yml

RUN mamba env update -n base -f ./environment.yml && \
    conda clean -afy

WORKDIR /root
RUN git clone https://github.com/Wiqzard/ochack.git /root/src


CMD ["bash"]
