'''
Tests for the vision_msgs builders in classifier_node.

Skipped entirely where ROS is not installed, so the rest of the suite still
runs on a plain Python environment. On a machine with ROS:

    source /opt/ros/noetic/setup.bash
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages python3 -m pytest tests/
'''

import os
import sys
import types

import pytest
from PIL import Image

pytest.importorskip('vision_msgs', reason='ROS vision_msgs not installed')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import triton_api  # noqa: E402
from fake_triton import FakeInferenceServerClient  # noqa: E402

BUS_JPG = os.path.join(HERE, 'assets', 'bus.jpg')
COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus']


@pytest.fixture(scope='module')
def node():
    '''
    Import classifier_node with the ROS runtime bits stubbed out.

    The builders under test are pure, but the module imports rospy/cv2/cv_bridge
    at import time and those need a master (or a working cv_bridge build) that a
    unit test should not require.
    '''
    saved = {name: sys.modules.get(name) for name in ('rospy', 'cv2', 'cv_bridge')}
    for name in ('rospy', 'cv2', 'cv_bridge'):
        sys.modules[name] = types.ModuleType(name)
    sys.modules['cv_bridge'].CvBridge = object
    sys.modules['cv2'].cvtColor = lambda *a: None
    sys.modules['cv2'].COLOR_BGR2RGB = 0
    try:
        import classifier_node
        yield classifier_node
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def image_msg():
    from std_msgs.msg import Header
    header = Header()
    header.frame_id = 'camera'
    return types.SimpleNamespace(header=header)


def bus_detections():
    client = FakeInferenceServerClient('yolov8n', response='bus')
    model = triton_api.Model(client, 'yolov8n')
    in_name = model.metadata.inputs[0].name
    out_name = model.metadata.outputs[0].name
    image_input = triton_api.ImageInput(
        scaling=triton_api.ScalingMode.NORM, layout='NCHW', letterbox=True)
    setattr(model, in_name, image_input)
    setattr(model, out_name,
            triton_api.DetectionOutput(confidence=0.25, labels=COCO))

    image = Image.open(BUS_JPG)
    return getattr(model.infer(image), out_name), image_input, image


def test_detections_become_a_detection2darray(node, image_msg):
    from vision_msgs.msg import Detection2DArray

    detections, image_input, image = bus_detections()
    array = node.build_detections(detections, image_input, image.size, image_msg)

    assert isinstance(array, Detection2DArray)
    assert array.header.frame_id == 'camera'
    assert len(array.detections) == len(detections)

    ids = [d.results[0].id for d in array.detections]
    assert 5 in ids and 0 in ids          # a bus and at least one person
    assert all(len(d.results) == 1 for d in array.detections)


def test_boxes_are_in_original_image_pixels(node, image_msg):
    '''The model saw a letterboxed 640x640; consumers need the photo's pixels.'''
    detections, image_input, image = bus_detections()
    array = node.build_detections(detections, image_input, image.size, image_msg)

    width, height = image.size          # 810x1080
    for detection in array.detections:
        bbox = detection.bbox
        assert 0 <= bbox.center.x <= width
        assert 0 <= bbox.center.y <= height
        assert bbox.size_x > 0 and bbox.size_y > 0
        assert bbox.center.theta == 0.0
        # and the box must fit inside the photo
        assert bbox.center.x - bbox.size_x / 2 >= -1
        assert bbox.center.x + bbox.size_x / 2 <= width + 1

    # The bus should be big in the source frame, not squeezed into 640px.
    bus = next(d for d in array.detections if d.results[0].id == 5)
    assert bus.bbox.size_x > 400


def test_detection2darray_serializes(node, image_msg):
    '''A message that cannot serialize would fail only at publish time.'''
    from io import BytesIO

    detections, image_input, image = bus_detections()
    array = node.build_detections(detections, image_input, image.size, image_msg)

    buffer = BytesIO()
    array.serialize(buffer)
    restored = type(array)()
    restored.deserialize(buffer.getvalue())

    assert len(restored.detections) == len(array.detections)
    assert restored.detections[0].results[0].id == array.detections[0].results[0].id


def test_classifications_become_a_classification2d(node, image_msg):
    from vision_msgs.msg import Classification2D

    client = FakeInferenceServerClient('mnist', response='gradient_top3')
    model = triton_api.Model(client, 'mnist')
    in_name = model.metadata.inputs[0].name
    out_name = model.metadata.outputs[0].name
    setattr(model, in_name, triton_api.ImageInput(layout='NCHW'))
    setattr(model, out_name, triton_api.ClassificationOutput(classes=3))

    results = getattr(model.infer(Image.new('L', (28, 28))), out_name)
    message = node.build_classification(results, image_msg)

    assert isinstance(message, Classification2D)
    assert len(message.results) == 3
    assert all(0 <= r.id <= 9 for r in message.results)
    scores = [r.score for r in message.results]
    assert scores == sorted(scores, reverse=True)
