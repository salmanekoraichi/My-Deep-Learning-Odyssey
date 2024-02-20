# base image
ARG PYTHON_VERSION=3.9
ARG docker_image_base=python:${PYTHON_VERSION}-slim
FROM ${docker_image_base}

# maintainers
LABEL maintainer1=soraya.arias@inria.fr maintainer2=jean-luc.parouty@simap.grenoble-inp.fr

ARG ARCH_VERSION=cpu
ARG BRANCH=pre-master

# Ensure a sane environment
ENV TZ=Europe/Paris LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update --fix-missing && \
    apt install -y --no-install-recommends apt-utils  \
        procps \
        python3-venv  \
        python3-pip && \
    apt -y dist-upgrade && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*  
   
# copy Python requirement packages list in docker image
COPY requirements-${ARCH_VERSION}.txt /root/requirements-${ARCH_VERSION}.txt

# Update Python tools and install requirements packages for Fidle
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir --upgrade -r /root/requirements-${ARCH_VERSION}.txt 

# Install tensorboard & update jupyter
RUN pip3 install --no-cache-dir --upgrade tensorboard tensorboardX jupyter ipywidgets

# Move default logo python
RUN bin/rm /usr/local/share/jupyter/kernels/python3/logo*      
    
# Change default logo and name kernels
COPY images/env-keras3.png /usr/local/share/jupyter/kernels/python3/logo-64x64.png
COPY images/env-keras3.svg /usr/local/share/jupyter/kernels/python3/logo-svg.svg

# Get Fidle datasets
RUN mkdir /data && \
    fid  install_datasets --quiet --install_dir /data

# Get Fidle notebooks and create link
RUN mkdir /notebooks/ && \
    fid install_notebooks --notebooks fidle-${BRANCH} --quiet --install_dir /notebooks && \
    ln -s $(ls -1td /notebooks/* | head -1) /notebooks/last

# Add Jupyter configuration (no browser, listen all interfaces, ...)
COPY jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py
COPY notebook.json /root/.jupyter/nbconfig/notebook.json

# Jupyter notebook uses 8888 
EXPOSE 8888

# tensorboard uses 6006 
EXPOSE 6006

VOLUME /notebooks
WORKDIR /notebooks

# Set Keras backend
ENV KERAS_BACKEND torch
# Set Python path to add fidle path
ENV PYTHONPATH=/notebooks/fidle-master/:$PYTHONPATH

# Set default shell (useful in the notebooks)
ENV SHELL=/bin/bash

# Set Fidle dataset directory variable
ENV FIDLE_DATASETS_DIR=/data/datasets-fidle

# Run a notebook by default
CMD ["jupyter", "lab"]