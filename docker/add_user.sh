#!/bin/bash

# set -e

if [ -z ${UID} ]; then 
  echo "Environment variable UID is not set. Please, export UID before executing docker-compose!"; 
fi

if [ -z ${ssh_password} ]; then 
  echo "ssh password is not set, setting it to default value."; 
  export ssh_password="3dreconstruction"
fi

mkdir -p $HOME

echo "Adding USER=$USER (UID=$UID HOME=$HOME) to container"
useradd -U -u $UID --shell /bin/bash -d $HOME $USER

# echo "Adding USER=$USER to sudoers in container"
# echo "$USER ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER
# chmod 0440 /etc/sudoers.d/$USER

cp -a /root/. $HOME/

groupmod -g $ORIG_USER_GID $USER
addgroup --gid $ORIG_DEV_GID developers
usermod -a -G $ORIG_DEV_GID $USER
addgroup --gid 1002594468 cube
usermod -a -G 1002594468 $USER

chown $USER $HOME
chown -R $USER $HOME/depth_prediction/depth_prediction

# password set for ssh access in port forwarding 
echo "Using port: ${sshd_port}"
sed -i "s/#Port 22/Port ${sshd_port}/" /etc/ssh/sshd_config
echo "${USER}:${ssh_password}" | chpasswd
service ssh start

echo "Starting new session and switching to USER=$USER"
su $USER

