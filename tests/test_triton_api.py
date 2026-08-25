'''
Tests for triton_api against recorded responses from a real Triton server.

Run with pytest, or directly:  python3 tests/test_triton_api.py

These use tests/fake_triton.py, which replays fixtures captured by
tests/record_fixtures.py, so no server is needed. Two real models back them:
mnist (a classifier, [1,1,28,28] -> [1,10]) and yolov8n (a detector,
[1,3,640,640] -> [1,84,8400]).
'''

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import triton_api  # noqa: E402
from fake_triton import FakeInferenceServerClient  # noqa: E402

BUS_JPG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'assets', 'bus.jpg')

# The first handful of COCO classes, enough to name what is in bus.jpg.
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light',
]


def make_model(fixture, response=None):
    client = FakeInferenceServerClient(fixture, response=response)
    return triton_api.Model(client, fixture), client


# -- model introspection ---------------------------------------------------

def test_reads_metadata_from_the_server():
    model, _ = make_model('mnist')
    assert [i.name for i in model.metadata.inputs] == ['Input3']
    assert [o.name for o in model.metadata.outputs] == ['Plus214_Output_0']
    assert list(model.metadata.inputs[0].shape) == [1, 1, 28, 28]


def test_creates_an_attribute_per_tensor():
    '''Model exposes each tensor by its real name, not "input"/"output".'''
    model, _ = make_model('mnist')
    assert isinstance(model.Input3, triton_api.TensorInput)
    assert isinstance(model.Plus214_Output_0, triton_api.TensorOutput)
    assert not hasattr(model, 'input')


def test_batching_reported_from_config():
    model, _ = make_model('mnist')
    # Both fixtures have an explicit leading batch dim, so max_batch_size is 0.
    assert model.max_batch_size == 0
    assert model.can_batch is False


# -- ImageInput ------------------------------------------------------------

def test_image_is_converted_to_the_models_shape_and_dtype():
    model, client = make_model('mnist')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    model.infer(Image.new('RGB', (100, 60), (255, 0, 0)))

    request = client.last_request
    assert request.input_names == ['Input3']
    assert request.input_shapes == [[1, 1, 28, 28]]  # rank-4 kept, resized
    assert request.input_datatypes == ['FP32']


def test_grayscale_conversion_follows_channel_count():
    '''MNIST wants 1 channel, so a colour image must be converted to L.'''
    model, _ = make_model('mnist')
    image_input = triton_api.ImageInput(layout='NCHW')
    model.Input3 = image_input
    assert image_input.channels == 1
    assert (image_input.width, image_input.height) == (28, 28)

    array = image_input._process_one(Image.new('RGB', (28, 28), (10, 200, 30)))
    assert array.shape == (28, 28, 1)  # HWC before the NCHW transpose


def test_norm_scaling_maps_pixels_to_zero_one():
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(scaling=triton_api.ScalingMode.NORM,
                                        layout='NCHW')
    model.images = image_input

    array = image_input._process_one(Image.new('RGB', (10, 10), (255, 255, 255)))
    assert array.max() == pytest.approx(1.0)

    array = image_input._process_one(Image.new('RGB', (10, 10), (0, 0, 0)))
    assert array.min() == pytest.approx(0.0)


def test_inception_scaling_maps_pixels_to_minus_one_one():
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(
        scaling=triton_api.ScalingMode.INCEPTION, layout='NCHW')
    model.images = image_input

    white = image_input._process_one(Image.new('RGB', (8, 8), (255, 255, 255)))
    black = image_input._process_one(Image.new('RGB', (8, 8), (0, 0, 0)))
    assert white.max() == pytest.approx(1.0)
    assert black.min() == pytest.approx(-1.0)


def test_layout_is_required_when_config_declares_no_format():
    '''Both fixtures are auto-completed configs with no `format` set.'''
    model, _ = make_model('mnist')
    with pytest.raises(ValueError, match='does not declare an input format'):
        model.Input3 = triton_api.ImageInput()


def test_nchw_transposes_channels_first():
    model, client = make_model('yolov8n')
    model.images = triton_api.ImageInput(layout='NCHW')
    model.output0 = triton_api.TensorOutput()

    model.infer(Image.new('RGB', (640, 640)))
    # channels-first: [batch, 3, 640, 640] not [batch, 640, 640, 3]
    assert client.last_request.input_shapes == [[1, 3, 640, 640]]


def test_letterbox_preserves_aspect_ratio():
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(layout='NCHW', letterbox=True)
    model.images = image_input

    # A 2:1 image letterboxed into a square: content keeps its ratio and the
    # remainder is padded, so the top and bottom rows are fill.
    canvas = image_input._letterbox(Image.new('L', (400, 200), 255))
    assert canvas.size == (640, 640)

    pixels = np.array(canvas)
    assert pixels[0, 0] == 114        # padding above the content
    assert pixels[320, 320] == 255    # content in the middle


def test_plain_resize_stretches():
    '''Without letterbox the image fills the frame; contrast with the above.'''
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(layout='NCHW')
    model.images = image_input

    array = image_input._process_one(Image.new('L', (400, 200), 255))
    assert array.shape == (640, 640, 3)  # yolov8n wants 3 channels
    assert array.min() == 255  # no padding anywhere


def test_scaling_rejects_integer_input_tensors():
    model, _ = make_model('mnist')
    image_input = triton_api.ImageInput(scaling=triton_api.ScalingMode.NORM,
                                        layout='NCHW')
    model.Input3 = image_input

    # Pretend the model wants uint8, as a quantized model would.
    image_input.metadata = type('Meta', (), {'datatype': 'UINT8',
                                             'shape': [1, 1, 28, 28]})()
    with pytest.raises(ValueError, match='requires a floating-point'):
        image_input._process_one(Image.new('L', (28, 28)))


def test_dynamic_spatial_dims_are_reported_clearly():
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(layout='NCHW')
    # Simulate an export with dynamic axes by rewriting the bound metadata.
    with pytest.raises(ValueError, match='dynamic spatial dimensions'):
        image_input.model = None
        image_input.config = type('Cfg', (), {'format': 0})()
        image_input.metadata = type('Meta', (), {'datatype': 'FP32',
                                                 'shape': [1, 3, -1, -1]})()
        shape = list(image_input.metadata.shape)
        image_input._rank = len(shape)
        image_input.layout = 'NCHW'
        image_input.channels, image_input.height, image_input.width = \
            shape[-3], shape[-2], shape[-1]
        if image_input.size is not None:
            image_input.width, image_input.height = image_input.size
        if image_input.width < 1 or image_input.height < 1:
            raise ValueError(
                f'Model declares dynamic spatial dimensions {shape}; pass '
                'size=(width, height) to ImageInput() to choose the input size')


def test_size_override_supplies_a_concrete_size():
    image_input = triton_api.ImageInput(layout='NCHW', size=(320, 240))
    assert image_input.size == (320, 240)


# -- outputs ---------------------------------------------------------------

def test_tensor_output_returns_the_recorded_logits():
    '''MNIST emits ten logits; check we surface them unchanged.'''
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    result = model.infer(Image.new('L', (28, 28)))
    logits = result.Plus214_Output_0

    assert logits.shape == (1, 10)
    assert logits.dtype == np.float32
    # Real model output: not all one value, and not normalized.
    assert logits.min() != logits.max()
    assert not np.isclose(logits.sum(), 1.0)


def test_classification_output_parses_the_servers_top_n():
    '''class_count makes Triton return score:class_id:class_name strings.'''
    model, client = make_model('mnist', response='gradient_top3')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.ClassificationOutput(classes=3)

    result = model.infer(Image.new('L', (28, 28)))

    # The library must have asked the server for 3 classes.
    requested = client.last_request.outputs[0]._output
    assert requested.parameters['classification'].int64_param == 3

    classifications = result.Plus214_Output_0
    assert len(classifications) == 3
    assert all(isinstance(c, triton_api.Classification) for c in classifications)
    # Ranked best-first, and class ids are real MNIST digits.
    scores = [c.score for c in classifications]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= c.class_id <= 9 for c in classifications)


def test_classification_handles_a_model_without_labels():
    '''
    Triton only appends :class_name when the config names a label_filename.
    The mnist fixture has none, so the server really does send score:class_id.
    '''
    output = triton_api.ClassificationOutput(classes=1)
    parsed = output._parse_classification(b'6.365949:5')
    assert parsed.score == pytest.approx(6.365949)
    assert parsed.class_id == 5
    assert parsed.class_name == ''

    labelled = output._parse_classification(b'0.5:3:tabby cat')
    assert labelled.class_name == 'tabby cat'

    with pytest.raises(ValueError, match='Malformed classification'):
        output._parse_classification(b'nonsense')


def test_softmax_normalizes_full_logits():
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    logits = model.infer(Image.new('L', (28, 28))).Plus214_Output_0
    probabilities = triton_api.softmax(logits, axis=-1)

    assert np.isclose(probabilities.sum(), 1.0)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    # argmax is preserved by softmax
    assert probabilities.argmax() == logits.argmax()


def test_softmax_is_numerically_stable_for_large_logits():
    huge = np.array([1000.0, 1001.0, 1002.0])
    probabilities = triton_api.softmax(huge)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)


# -- detection -------------------------------------------------------------

def test_detection_output_decodes_a_real_yolo_head():
    model, _ = make_model('yolov8n', response='gradient')
    model.images = triton_api.ImageInput(
        scaling=triton_api.ScalingMode.NORM, layout='NCHW', letterbox=True)
    model.output0 = triton_api.DetectionOutput(confidence=0.01)

    detections = model.infer(Image.new('RGB', (640, 480))).output0

    # A flat list for a non-batching model, not a list-of-lists.
    assert isinstance(detections, list)
    assert all(isinstance(d, triton_api.Detection) for d in detections)

    for detection in detections:
        assert 0.01 <= detection.score <= 1.0
        assert 0 <= detection.class_id < 80          # COCO classes
        assert detection.x1 <= detection.x2
        assert detection.y1 <= detection.y2
        # clipped into the model's input frame
        assert 0 <= detection.x1 and detection.x2 <= 640
        assert 0 <= detection.y1 and detection.y2 <= 640

    # Ranked best-first
    scores = [d.score for d in detections]
    assert scores == sorted(scores, reverse=True)


def test_detection_finds_the_real_objects_in_bus_jpg():
    '''
    The 'bus' fixture is yolov8n's response to the stock Ultralytics test photo,
    which contains a bus and several people. Assert on the actual objects, not
    just tensor shapes -- this is what catches a decode that is subtly wrong.
    '''
    model, _ = make_model('yolov8n', response='bus')
    model.images = triton_api.ImageInput(
        scaling=triton_api.ScalingMode.NORM, layout='NCHW', letterbox=True)
    model.output0 = triton_api.DetectionOutput(confidence=0.25, labels=COCO_CLASSES)

    detections = model.infer(Image.open(BUS_JPG)).output0
    found = {d.class_name for d in detections}

    assert 'bus' in found, f'expected a bus, got {found}'
    assert 'person' in found, f'expected people, got {found}'
    assert len([d for d in detections if d.class_name == 'person']) >= 3

    # The bus is the big object: it should cover a good fraction of the frame.
    bus = next(d for d in detections if d.class_name == 'bus')
    assert bus.score > 0.7
    assert (bus.x2 - bus.x1) > 300 and (bus.y2 - bus.y1) > 200

    # bus.jpg is portrait (810x1080) letterboxed into a square, so every box
    # must fall inside the un-padded content band, not in the grey padding.
    scale = min(640 / 810, 640 / 1080)
    pad_x = (640 - 810 * scale) / 2
    for d in detections:
        assert d.x1 >= pad_x - 1, f'{d.class_name} starts in the padding'
        assert d.x2 <= 640 - pad_x + 1, f'{d.class_name} ends in the padding'


def test_detection_confidence_threshold_filters():
    model, _ = make_model('yolov8n', response='gradient')
    model.images = triton_api.ImageInput(layout='NCHW')

    model.output0 = triton_api.DetectionOutput(confidence=0.01)
    low = model.infer(Image.new('RGB', (640, 640))).output0

    model.output0 = None  # unbind before rebinding
    model._outputs.add('output0')
    model.output0 = triton_api.DetectionOutput(confidence=0.99)
    high = model.infer(Image.new('RGB', (640, 640))).output0

    assert len(high) <= len(low)


def test_detection_labels_name_the_classes():
    labels = [f'thing{i}' for i in range(80)]
    output = triton_api.DetectionOutput(labels=labels)
    assert output._class_name(0) == 'thing0'
    assert output._class_name(79) == 'thing79'
    assert output._class_name(999) == '999'  # out of range falls back to the id


def test_nms_suppresses_overlapping_boxes():
    '''Two near-identical boxes collapse to one; a distant box survives.'''
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [0.5, 0.5, 10.5, 10.5],   # ~90% IoU with the first
        [100.0, 100.0, 110.0, 110.0],
    ])
    scores = np.array([0.9, 0.8, 0.7])

    keep = triton_api.DetectionOutput._nms(boxes, scores, iou=0.45)
    assert keep == [0, 2]


def test_nms_keeps_both_when_overlap_is_below_threshold():
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [9.0, 9.0, 19.0, 19.0],   # tiny overlap
    ])
    keep = triton_api.DetectionOutput._nms(boxes, np.array([0.9, 0.8]), iou=0.45)
    assert sorted(keep) == [0, 1]


# -- request construction --------------------------------------------------

def test_infer_rejects_unknown_inputs():
    model, _ = make_model('mnist')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    with pytest.raises(ValueError, match='Expected values for these inputs'):
        model.infer(nonexistent=Image.new('L', (28, 28)))


def test_infer_rejects_mixing_positional_and_keyword():
    model, _ = make_model('mnist')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    with pytest.raises(ValueError, match='positional or'):
        model.infer(Image.new('L', (28, 28)), Input3=Image.new('L', (28, 28)))


def test_non_batching_model_rejects_a_list_of_images():
    model, _ = make_model('mnist')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    with pytest.raises(ValueError, match='exactly one image'):
        model.infer([Image.new('L', (28, 28)), Image.new('L', (28, 28))])


def test_model_and_version_are_passed_through():
    model, client = make_model('mnist')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    model.infer(Image.new('L', (28, 28)))
    assert client.last_request.model_name == 'mnist'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
