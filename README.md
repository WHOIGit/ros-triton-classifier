# Triton Inference Client for Image Classification in ROS

This package provides a ROS node that submits images to a [Triton Inference Server][triton] instance.

The message definitions are similar to [`vision_msgs`][vision_msgs] but it does not currently adhere to that spec.

  [triton]: https://developer.nvidia.com/nvidia-triton-inference-server
  [vision_msgs]: https://github.com/ros-perception/vision_msgs


## Installation

Please install the `tritonclient[all]` Python package:

    python3 -m pip install tritonclient[all]


## Parameters

- `~triton_server_url` (required): Triton Inference Server URL.
- `~classifier_model` (required): Triton model name to use for classification.
- `~image_topic` (required): Base image topic. The node subscribes to `<image_topic>/raw`.
- `~classification_topic` (optional): Classification output topic. Defaults to `<image_topic>/class`.
