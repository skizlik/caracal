# Use verified NVIDIA CUDA image that matches your system
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    sudo \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install additional CUDA runtime libraries that TensorFlow needs
RUN apt-get update && apt-get install -y \
    cuda-runtime-12-9 \
    libcudnn8 \
    libcublas-12-9 \
    libcufft-12-9 \
    libcurand-12-9 \
    libcusolver-12-9 \
    libcusparse-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Test Python version
RUN python --version

# Install your library's core dependencies
RUN pip install \
    tensorflow==2.16.1 \
    jupyterlab \
    poetry

# Install ML packages your notebook needs
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    pillow

# Install your library's other dependencies
RUN pip install \
    hyperopt \
    shap \
    mlflow

# Set CUDA environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_CACHE_DISABLE=1
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"


# Working directory
WORKDIR /home/appuser/projects

CMD ["/bin/bash"]
