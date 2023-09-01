# NVIDIA does not publish a Triton container image for Jetson devices.
# https://github.com/triton-inference-server/server/issues/4781
#
# This unofficial container image unpacks the Triton for Jetson tarball onto a 
# L4T base image along with dependencies documented here:
# https://github.com/triton-inference-server/server/blob/main/docs/user_guide/jetson.md
#
# Example usage:
#
#   docker build -t tritonserver:22.02-pyt-python-py3 -f Dockerfile.jetson .
#   docker run --rm --name triton --runtime nvidia \
#     --volume /usr/local/share/triton/models:/models \
#     --publish 8000:8000/tcp \
#     --publish 8001:8001/tcp \
#     --publish 8002:8002/tcp \
#     tritonserver:22.02-pyt-python-py3 \
#     tritonserver --model-repository /models
#
# There are other base images (l4t-ml, l4t-pytorch, etc.) that might be better.

FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Update CA certificates so that we don't get TLS issues downloading from GitHub
RUN apt update \
 && apt install -y \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Uncomment if you have pre-downloaded the Triton sources
# COPY tritonserver.tgz .

# Download and unpack the Triton sources
RUN ([ -f tritonserver.tgz ] || wget -O tritonserver.tgz \
        https://github.com/triton-inference-server/server/releases/download/v2.19.0/tritonserver2.19.0-jetpack4.6.1.tgz) \
 && mkdir /opt/tritonserver \
 && tar xf tritonserver*.tgz -C /opt/tritonserver --strip-components 1 \
 && rm -f tritonserver*.tgz

# NOTE:
# The following instructions are based on docs/jetson.md, and based on the work
# of @CoderHam to identify only the runtime dependencies.
RUN apt update \
 && apt install -y \
        libarchive-dev \
        libb64-0d \
        libomp5 \
        libomp-dev \
        libopenblas-dev \
        libre2-4 \
        libssl1.1 \
        python3 \
        python3-dev \
        python3-pip \
        rapidjson-dev \
        zlib1g \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip first, because otherwise we can't install aiohttp
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --upgrade \
        aiohttp \
        expecttest \
        hypothesis \
        ninja \
        protobuf \
        pyyaml \
        scipy \
        typing_extensions \
        xmlrunner \
 && python3 -m pip install --upgrade \
        https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl

# You cannot append to an environment variable in a Dockerfile, so we need to
# set the whole value, which hardcodes our CUDA version.
ENV PATH /opt/tritonserver/bin:/usr/local/cuda-10.2/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# As of libnvidia-container v0.11.0+jetpack, the CUDA libraries from the host
# are not exposed to the container unless we set this environment variable.
ENV NVIDIA_REQUIRE_JETPACK csv-mounts=all

# HTTP API, gRPC API, and Prometheus metrics, respectively
EXPOSE 8000 8001 8002
