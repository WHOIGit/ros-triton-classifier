# Triton Inference Client for Image Classification in ROS

This package provides a ROS node that submits images to a [Triton Inference Server][triton] instance.

The message definitions are similar to [`vision_msgs`][vision_msgs] but it does not currently adhere to that spec.

  [triton]: https://developer.nvidia.com/nvidia-triton-inference-server
  [vision_msgs]: https://github.com/ros-perception/vision_msgs


## Installation

Please install the `tritonclient[all]` Python package:

    python3 -m pip install tritonclient[all]

### Server (Jetson)

`triton/Dockerfile.jetson` builds a Triton Inference Server image for Jetson.
The host must run L4T r35.4.1 or newer (JetPack 5.1.2+); see the comments in
that file for the rationale. Build and run with:

    docker build -t triton-jetson:2.35.0-jp512 -f triton/Dockerfile.jetson triton
    docker run --rm --runtime nvidia --ipc=host \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        -v /path/to/models:/models \
        triton-jetson:2.35.0-jp512

By default the container runs `tritonserver --model-repository=/models`, so
bind-mount your model repository at `/models` (or append your own
`tritonserver ...` command to the `docker run` line to override it).

The image supports the TensorRT, ONNX Runtime, and Python backends only; the
old image's PyTorch/libtorch provisioning was intentionally dropped, so
TorchScript models will not load.

Build arguments (pass with `docker build --build-arg`): `TRITON_VERSION` and
`JETPACK_VERSION` select the Triton release (defaults 2.35.0 / 5.1.2), from
which `TRITON_TARBALL` and `TRITON_URL` are derived (override them directly
for a differently named asset or mirror), and `TRITON_SHA256` optionally pins
the tarball's checksum. You can also pre-place the tarball in `triton/` to
skip the download.
