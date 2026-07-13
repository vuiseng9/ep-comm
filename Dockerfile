FROM nvcr.io/nvidia/pytorch:26.04-py3
SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update && \
    apt-get install -y \
        bash-completion \
        git-lfs \
        wget \
        build-essential \
        gdb && \
    rm -rf /var/lib/apt/lists/*


RUN cat << 'EOF' >> ~/.bashrc
if [ -f /etc/bash_completion ]; then
  . /etc/bash_completion
fi
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\] : \[\033[01;34m\]\w\[\033[01;31m\]$(__git_ps1)\[\033[00m\]\n$ '
EOF

ARG REPO=https://github.com/vuiseng9/symmem-ep
ARG BRANCH=main

WORKDIR /workspace
RUN git clone $REPO && cd symmem-ep && make install-dep

WORKDIR /workspace/symmem-ep
CMD ["bash"]
