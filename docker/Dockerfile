FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ARG no_proxy
ARG http_proxy
ARG https_proxy

# Install some base apps
RUN apt-get update 
RUN apt-get install -y openssh-server wget nano libgl1-mesa-glx

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# add tini
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

WORKDIR /train/code

# Add conf files
COPY conf_files/depth_prediction_conda.yml conf_files/depth_prediction_conda.yml

# Create the conda env
RUN conda env create -f conf_files/depth_prediction_conda.yml &&  \
    echo "conda activate depth_prediction_conda" >> ~/.bashrc

# Copy the entrypoint for working on local machine
COPY add_user.sh /add_user.sh
RUN chmod +x /add_user.sh

# Add write permission to some folders 
RUN chmod 777 /train && \
    chmod 777 /opt/conda/envs/depth_prediction_conda && \
    mkdir -p /.conda/envs/ && touch /.conda/envs/.conda_envs_dir_test && chmod 777 /.conda/envs/.conda_envs_dir_test

COPY . .

# Steps to enable pycharm usage with the docker
RUN mkdir /var/run/sshd

RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
# RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

RUN echo "export PYTHONAPTH=/train/code" >> /etc/profile

# openh symbolic link for opencv to work
RUN ln -s /opt/conda/envs/depth_prediction_conda/lib/libopenh264.so /opt/conda/envs/depth_prediction_conda/lib/libopenh264.so.5
