#
#   ref https://github.com/tebeka/pythonwise/blob/master/docker-miniconda/Dockerfile
#
#   miniconda vers: http://repo.continuum.io/miniconda
#   sample variations:
#     Miniconda3-latest-Linux-armv7l.sh
#     Miniconda3-latest-Linux-x86_64.sh
#     Miniconda3-py38_4.10.3-Linux-x86_64.sh
#     Miniconda3-py37_4.10.3-Linux-x86_64.sh
#
#   py vers: https://anaconda.org/anaconda/python/files
#   tf vers: https://anaconda.org/anaconda/tensorflow/files
#   tf-mkl vers: https://anaconda.org/anaconda/tensorflow-mkl/files
#

ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10

FROM ubuntu:${UBUNTU_VER}

# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
# RUN conda create --name=prime python=3.10
# Make RUN commands use the new environment:
# RUN echo "conda activate prime" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -c conda-forge graph-tool
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends apt-utils build-essential sudo git
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
WORKDIR /whooshai

COPY ./whooshai/ whooshai
COPY .env .env
COPY makefile makefile
COPY ./weights/* ./weights
COPY ./data/ind.whoosh.graph data/ind.whoosh.graph
COPY ./data/ind.whoosh.x data/ind.whoosh.x
COPY ./data/ind.whoosh.y data/ind.whoosh.y
COPY ./data/ind.whoosh.tx data/ind.whoosh.tx
COPY ./data/ind.whoosh.ty data/ind.whoosh.ty
COPY ./data/ind.whoosh.ty data/ind.whoosh.test.index
COPY ./requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==2.0.1
RUN pip install torch-scatter==2.1.1
RUN pip install torch-sparse==0.6.17
RUN pip install torch-geometric==2.3.1
RUN pip install -r requirements.txt
ENV WORKERS_PER_CORE=3

CMD ["make", "run"]
