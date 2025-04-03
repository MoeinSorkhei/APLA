FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ARG UID
ARG GID
ARG USER
ARG GROUP
ARG WANDB_KEY
ARG GITHUB_EMAIL
ARG GITHUB_USERNAME

SHELL [ "/bin/bash", "--login", "-c" ]

# handle recent nvidia key rotation happened in Apr. 2022 (see here: https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install utilities (xdg-utils required by NBIA)
RUN apt-get update && apt-get install -y --no-install-recommends \
        file \
        xdg-utils \
        rsync \
        tree \
        unrar \
        man \
        build-essential \
        apt-utils \
        screen \
        cmake \
        nano \
        bc \
        jq \
        git \
        curl \
        ca-certificates \
        sudo \
        bzip2 \
        libx11-6 \
        git \
        wget \
        ssh-client \
        libjpeg-dev \
        bash-completion \
        libgl1-mesa-dev \
        ffmpeg \
        tmux \
        htop \
        nfs-common \
        cifs-utils \
        zip \
        unzip \
        pydf \
        nnn \
        ncdu \
        aria2 \
        mdadm \
        net-tools \
        uidmap \
        openslide-tools \
        libjemalloc-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ENV HOME=/home/$USER

RUN addgroup --gid $GID $GROUP \
    && adduser --disabled-password \
    --gecos "" \
    --uid $UID \
    --gid $GID \
    --shell /bin/bash \
    --home $HOME \
    $USER 
WORKDIR $HOME


# getting NBIA for TCIA
RUN wget "https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.2/nbia-data-retriever-4.2.deb" \
    -P /tmp/
RUN mkdir /usr/share/desktop-directories  # for xdg-utils
RUN sudo -S dpkg -i /tmp/nbia-data-retriever-4.2.deb


# install bat
RUN wget https://github.com/sharkdp/bat/releases/download/v0.20.0/bat-musl_0.20.0_amd64.deb -P /tmp/
RUN sudo dpkg -i /tmp/bat-musl_0.20.0_amd64.deb


# install bottom (seeing CPU usage, processes etc.) -- usage: btm (https://github.com/ClementTsang/bottom)
RUN curl -L https://github.com/ClementTsang/bottom/releases/download/0.6.8/bottom_0.6.8_amd64.deb --output /tmp/bottom_0.6.8_amd64.deb
RUN sudo dpkg -i /tmp/bottom_0.6.8_amd64.deb


# install miniconda (if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3)
ENV MINICONDA_VERSION=py38_4.8.3

ENV CONDA_DIR=/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# build the conda environment
ENV ENV_PREFIX=$HOME/env
RUN conda update --name base --channel defaults conda && \
conda install python=3.8 \
        pytorch torchvision cudatoolkit=11.3 \
        scikit-learn \
        easydict \
        matplotlib \
        tqdm \
        ipykernel \
        pandas \
        ipywidgets \
        umap-learn \
        scikit-image \
        gdcm \
        -c pytorch -c conda-forge -c pyviz -y \
    && conda clean --all --yes

# most recent wand version 0.12 breaks the program, and setting the version in the previous conda install takes a day to solve the environment
RUN conda install -c conda-forge wandb==0.11.0

# pip installations
ENV SHELL=/bin/bash
RUN pip install jupyter jupyterlab \
    tabulate colorcet datashader bokeh holoviews h5py tensorboard termcolor \
    timm seaborn kaggle zenodo-get gdown pydicom opencv-python nibabel tldr

# other manual pip installations that I did inside my container:
### pip install pyemd einops munkres geomloss
### pip install pyemd
### pip install einops
### pip install munkres
### pip install geomloss


# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# RUN wandb login
ENV WANDB_API_KEY=$WANDB_KEY

# Authorize SSH Host and Add the keys and set permissions for GIT
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts &&\
    echo "$RSA_PRIV_KEY" > /root/.ssh/id_rsa && \
    echo "$RSA_PUB_KEY" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub && \
    git config --global user.email "$GITHUB_EMAIL" && \
    git config --global user.name "$GITHUB_USERNAME"

CMD ["/bin/bash"]
