# Use the official NVIDIA CUDA image with Ubuntu as the base
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Set environment variables to make apt-get non-interactive
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    CUDA_HOME=/usr/local/cuda-11.4 \
    PATH=/usr/local/cuda-11.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

    RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}')" \
    && echo "PyTorch and CUDA versions successfully checked."


# Set working directory
WORKDIR /usr/app

# Update and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    wget \
    curl \
    vim \
    libx11-6 \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*


RUN pip install gdown open3d werkzeug==3.0.4 pygad shapely==1.8.1 tqdm pandas==1.3.4 numba==0.49.0 timm  opencv-python==3.4.18.65 matplotlib~=3.3.3 scipy~=1.5.4 --ignore-installed llvmlite==0.32.1 backports.tarfile
RUN pip install --upgrade flask dash
RUN apt-get update
RUN apt-get install -y libx11-6 libgl1-mesa-glx libglib2.0-0

# Set the default command to bash

RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}')" \
    && echo "PyTorch and CUDA versions successfully checked."

CMD ["bash"]
