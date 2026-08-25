# Triton Inference Client for Image Classification in ROS

This package provides a ROS node that submits images to a [Triton Inference Server][triton] instance.

The message definitions are similar to [`vision_msgs`][vision_msgs] but it does not currently adhere to that spec.

  [triton]: https://developer.nvidia.com/nvidia-triton-inference-server
  [vision_msgs]: https://github.com/ros-perception/vision_msgs


## Installation

Please install the `tritonclient[all]` Python package:

    python3 -m pip install tritonclient[all]

This client uses the gRPC transport only. On Jetson (JetPack 5.1.x / aarch64,
Python 3.8) the `[all]` and `[http]` extras fail to build because they pull in
`gevent`, which has no aarch64 wheel; install just the gRPC client there:

    python3 -m pip install tritonclient[grpc]

### Server (Jetson)

`triton/Dockerfile.jetson` builds a Triton Inference Server image for JetPack
5.1.x (L4T r35.x); see the comments in that file. Build and run with:

    docker build -t triton-jetson:2.35.0-jp512 -f triton/Dockerfile.jetson triton
    docker run --rm --runtime nvidia --ipc=host \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        -v /path/to/models:/models \
        triton-jetson:2.35.0-jp512 tritonserver --model-repository=/models
