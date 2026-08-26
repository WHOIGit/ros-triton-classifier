# Triton Inference Client for ROS

This package provides a ROS node that submits images to a [Triton Inference
Server][triton] instance and publishes the results as [`vision_msgs`][vision_msgs].

  [triton]: https://developer.nvidia.com/nvidia-triton-inference-server
  [vision_msgs]: https://github.com/ros-perception/vision_msgs

## Topics

The node subscribes to `~image_topic` (the base topic — `image_transport`
publishes the raw `sensor_msgs/Image` there, and gives subtopics only to its
other transports) and publishes, depending on `~task`:

| `~task` | topic | type |
| --- | --- | --- |
| `classification` | `<image_topic>/classification` | `vision_msgs/Classification2D` |
| `detection` | `<image_topic>/detections` | `vision_msgs/Detection2DArray` |
| either | `<image_topic>/vision_info` | `vision_msgs/VisionInfo` (latched) |

Detection boxes are in the source image's pixels, not the model's input frame.

## Class names

`vision_msgs` identifies a class by a numeric `id` only, so names travel out of
band. Give the node its labels with either:

    ~class_labels: [amphipod, appendicularian, chaetognath, ...]   # a list
    ~class_labels_file: /path/to/labels.txt                        # one per line

The node loads them, puts them on the parameter server as an id → name map, and
publishes a latched `VisionInfo` whose `database_location` is the resolved
parameter name. Consumers read `VisionInfo` once and then look the map up:

```python
info = rospy.wait_for_message('<image_topic>/vision_info', VisionInfo)
if info.database_location:      # empty when the node has no labels configured
    labels = rospy.get_param(info.database_location)   # {'0': 'amphipod', ...}
    name = labels[str(detection.results[0].id)]
```

`database_version` changes whenever the model or its labels change, so a
consumer can cache the map and re-read it only when the version moves.

This indirection is what `VisionInfo` exists for — its own definition
recommends storing the database "as an XML string on the ROS parameter server"
— and it follows the convention used by [`ros_deep_learning`][rdl]. Labels are
optional; without them the node publishes numeric ids and logs a warning.

  [rdl]: https://github.com/dusty-nv/ros_deep_learning

### Other parameters

`~triton_server_url`, `~classifier_model`, `~layout` (`NCHW`/`NHWC`, needed when
the model config declares no `format`), `~input_size` (`[width, height]`, needed
when the model declares dynamic spatial dimensions), `~classes`
(classification), `~confidence_threshold` and `~iou_threshold` (detection).


## Installation

Please install the `tritonclient[all]` Python package:

    python3 -m pip install tritonclient[all]

This client uses the gRPC transport only. On Jetson (JetPack 5.1.x / aarch64,
Python 3.8) the `[all]` and `[http]` extras fail to build because they pull in
`gevent`, which has no aarch64 wheel; install just the gRPC client there:

    python3 -m pip install tritonclient[grpc]

## Tests

    python3 -m pytest tests/

The tests need no Triton server. `tests/fixtures/` holds real gRPC responses
recorded from a live server for two small models -- `mnist` (a classifier,
`[1,1,28,28]` -> `[1,10]`) and `yolov8n` (a detector, `[1,3,640,640]` ->
`[1,84,8400]`) -- and `tests/fake_triton.py` replays those recorded protobufs
through the same client types the real library returns. So the tests exercise
how `triton_api` builds requests and interprets responses, against responses a
server actually produced, offline and in under a second.

To re-record after changing the models or the Triton version, serve `mnist` and
`yolov8n` and run:

    python3 tests/record_fixtures.py -u localhost:8001
