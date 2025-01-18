# Start with Ubuntu 22.04 base image
FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies and build tools (gcc, g++, make, etc.)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    software-properties-common \
    build-essential \
    gcc \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    cmake \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*


# Add the deadsnakes PPA to install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Fetch and install pip for Python 3.10 using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set Python 3.10 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Install python3.10-distutils to get distutils.msvccompiler
RUN apt-get update && apt-get install -y \
    python3.10-distutils \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Cython, wheel, and setuptools
RUN pip install --no-cache-dir "Cython==0.29.21" wheel setuptools

# Install PyYAML separately
RUN pip install --no-cache-dir "pyyaml==5.4.1" --no-build-isolation

# Install NumPy separately
RUN pip install --no-cache-dir "numpy==1.26.4"

#Install Graphviz
RUN apt-get update && apt-get install -y graphviz libgraphviz-dev

#Install DGL, torch and its friends. :)))
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

#Install Java
RUN apt-get update && apt install -y default-jre

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# set DGL environment variables
RUN export DGLBACKEND=pytorch

# Default command
CMD ["/bin/bash"]

# EXPOSE 6006