FROM eu.artifactory.conti.de:6009/rose_base:1.2.0

ARG no_proxy
ARG http_proxy
ARG https_proxy

COPY . /train/static_code

# install miniconda
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

WORKDIR /train/static_code

# Create the conda env
RUN conda env create -f docker/conf_files/metalearning_pytorch.yml &&  \
    echo "conda activate metalearning_pytorch" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "rose_conda_pytorch", "/bin/bash", "-c"]

# Copy the entrypoint for working on local machine
COPY docker/add_user.sh /add_user.sh
RUN chmod +x /add_user.sh

# Copy the entrypoint for working on cluster
COPY docker/entrypoint.sh /entrypoint
RUN chmod +x /entrypoint

# Add write permission to some folders 
RUN chmod 777 /train && \
    chmod 777 /opt/conda/envs/rose_conda_pytorch && \
    mkdir -p /.conda/envs/ && touch /.conda/envs/.conda_envs_dir_test && chmod 777 /.conda/envs/.conda_envs_dir_test

# Steps to enable pycharm usage with the docker
# Install openssh to enable ssh 
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
# RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

RUN echo "export PYTHONAPTH=/train/code" >> /etc/profile

CMD /entrypoint
