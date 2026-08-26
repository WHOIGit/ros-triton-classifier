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

# All 80 COCO classes, in id order. DetectionOutput checks the label count
# against the model's class axis, so a truncated list is rejected.
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
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


def test_source_mapping_round_trips_a_letterboxed_box():
    '''Detections come back in input space; check we can undo the letterbox.'''
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(layout='NCHW', letterbox=True)
    model.images = image_input

    source = (810, 1080)  # bus.jpg: portrait into a square model
    scale_x, scale_y, pad_x, pad_y = image_input.source_mapping(source)
    assert scale_x == scale_y                      # aspect preserved
    assert pad_y == 0 and pad_x > 0                # padded left/right only

    # A box covering the whole source maps to the whole content band and back.
    box = (pad_x, pad_y, 640 - pad_x, 640 - pad_y)
    x1, y1, x2, y2 = image_input.to_source_box(box, source)
    assert x1 == pytest.approx(0, abs=1)
    assert y1 == pytest.approx(0, abs=1)
    assert x2 == pytest.approx(810, abs=1)
    assert y2 == pytest.approx(1080, abs=1)


def test_source_mapping_without_letterbox_uses_two_scales():
    model, _ = make_model('yolov8n')
    image_input = triton_api.ImageInput(layout='NCHW')  # stretch
    model.images = image_input

    scale_x, scale_y, pad_x, pad_y = image_input.source_mapping((1280, 640))
    assert (pad_x, pad_y) == (0.0, 0.0)
    assert scale_x == pytest.approx(0.5)   # 640/1280
    assert scale_y == pytest.approx(1.0)   # 640/640


def test_bus_detections_map_back_onto_the_original_photo():
    model, _ = make_model('yolov8n', response='bus')
    image_input = triton_api.ImageInput(
        scaling=triton_api.ScalingMode.NORM, layout='NCHW', letterbox=True)
    model.images = image_input
    model.output0 = triton_api.DetectionOutput(confidence=0.25,
                                               labels=COCO_CLASSES)

    image = Image.open(BUS_JPG)
    detections = model.infer(image).output0
    bus = next(d for d in detections if d.class_name == 'bus')

    x1, y1, x2, y2 = image_input.to_source_box((bus.x1, bus.y1, bus.x2, bus.y2),
                                               image.size)
    # Back in the photo's own 810x1080 pixels, and still a big bus-shaped box.
    assert 0 <= x1 < x2 <= 810
    assert 0 <= y1 < y2 <= 1080
    assert (x2 - x1) > 400 and (y2 - y1) > 300


def test_scaling_rejects_integer_input_tensors():
    '''A scaling mode bound to an integer tensor must fail at bind time.'''
    client = FakeInferenceServerClient('mnist')
    # Pretend the model wants uint8, as a quantized model would.
    client.metadata.inputs[0].datatype = 'UINT8'
    model = triton_api.Model(client, 'mnist')

    with pytest.raises(ValueError, match='requires a floating-point'):
        model.Input3 = triton_api.ImageInput(
            scaling=triton_api.ScalingMode.NORM, layout='NCHW')


def test_dynamic_spatial_dims_are_reported_clearly():
    '''Binding to a dynamic-axes export must raise the real library error.'''
    client = FakeInferenceServerClient('yolov8n')
    client.metadata.inputs[0].shape[:] = [1, 3, -1, -1]
    model = triton_api.Model(client, 'yolov8n')

    with pytest.raises(ValueError, match='dynamic spatial dimensions'):
        model.images = triton_api.ImageInput(layout='NCHW')


def test_size_override_supplies_a_concrete_size():
    '''size= fills in the dynamic dimensions and makes bind() succeed.'''
    client = FakeInferenceServerClient('yolov8n')
    client.metadata.inputs[0].shape[:] = [1, 3, -1, -1]
    model = triton_api.Model(client, 'yolov8n')

    image_input = triton_api.ImageInput(layout='NCHW', size=(320, 240))
    model.images = image_input
    assert (image_input.width, image_input.height) == (320, 240)


def test_size_override_must_agree_with_fixed_dims():
    '''size= cannot silently override a size the model declares as fixed.'''
    model, _ = make_model('yolov8n')
    with pytest.raises(ValueError, match='contradicts'):
        model.images = triton_api.ImageInput(layout='NCHW', size=(320, 240))


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


def test_softmax_activation_does_not_let_the_server_truncate():
    '''The whole point: an activation needs every logit, so class_count is 0.'''
    model, client = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.ClassificationOutput(
        classes=3, activation='softmax')

    model.infer(Image.new('L', (28, 28)))

    requested = client.last_request.outputs[0]._output
    assert requested.parameters['classification'].int64_param == 0


def test_softmax_activation_yields_real_probabilities():
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    # Ask for all ten so we can check they form a distribution.
    model.Plus214_Output_0 = triton_api.ClassificationOutput(
        classes=10, activation='softmax')

    results = model.infer(Image.new('L', (28, 28))).Plus214_Output_0

    assert len(results) == 10
    scores = [c.score for c in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert sum(scores) == pytest.approx(1.0)       # over the FULL class set
    assert scores == sorted(scores, reverse=True)  # ranked best-first
    assert len({c.class_id for c in results}) == 10


def test_truncated_softmax_would_have_been_wrong():
    '''
    Demonstrates issue #5. Normalizing the server's top-3 overstates every
    confidence, because the other seven logits are missing from the sum.
    '''
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.ClassificationOutput(
        classes=10, activation='softmax')
    correct = model.infer(Image.new('L', (28, 28))).Plus214_Output_0

    # What you would get by softmaxing only the top 3 scores.
    top3_logits = np.array([c.score for c in correct[:3]])
    naive = triton_api.softmax(np.log(top3_logits))   # renormalize the subset

    assert naive[0] > correct[0].score               # inflated
    assert sum(naive) == pytest.approx(1.0)          # sums to 1 over a subset
    assert sum(c.score for c in correct[:3]) < 1.0   # honest: leaves room


def test_sigmoid_activation_is_independent_per_class():
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.ClassificationOutput(
        classes=10, activation='sigmoid')

    results = model.infer(Image.new('L', (28, 28))).Plus214_Output_0
    scores = [c.score for c in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
    # multi-label: no requirement to sum to 1
    assert sum(scores) != pytest.approx(1.0)


def test_activation_labels_name_the_classes():
    digits = [f'digit-{i}' for i in range(10)]
    model, _ = make_model('mnist', response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.ClassificationOutput(
        classes=2, activation='softmax', labels=digits)

    results = model.infer(Image.new('L', (28, 28))).Plus214_Output_0
    for c in results:
        assert c.class_name == f'digit-{c.class_id}'


def test_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match='activation must be one of'):
        triton_api.ClassificationOutput(classes=3, activation='relu')


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

    # Plain reassignment rebinds an output; no private state involved.
    model.output0 = triton_api.DetectionOutput(confidence=0.99)
    high = model.infer(Image.new('RGB', (640, 640))).output0

    # A permissive threshold must find something and a strict one must
    # actually discard some of it, or this test cannot detect a broken
    # filter (0 <= 0 would pass a filter that drops everything).
    assert len(low) > 0
    assert len(high) < len(low)


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
    with pytest.raises(ValueError, match='at most 1 image'):
        model.infer([Image.new('L', (28, 28)), Image.new('L', (28, 28))])


def test_model_and_version_are_passed_through():
    client = FakeInferenceServerClient('mnist', response='gradient',
                                       model_version='7')
    model = triton_api.Model(client, 'mnist', version='7')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    model.infer(Image.new('L', (28, 28)))
    # The fake raises on any name/version mismatch, so reaching these
    # assertions already proves config/metadata carried them too.
    assert client.last_request.model_name == 'mnist'
    assert client.last_request.model_version == '7'


def test_fake_validates_the_requested_model_name():
    client = FakeInferenceServerClient('mnist')
    with pytest.raises(AssertionError, match='serves'):
        triton_api.Model(client, 'some_other_model')


# -- batching models -------------------------------------------------------

def make_batched_mnist(response=None):
    '''
    No fixture was recorded from a batch-enabled model, so derive one: give
    mnist a max_batch_size and the implicit leading batch axis Triton reports
    for such models. The recorded single-image responses replay unchanged as
    batches of one.
    '''
    client = FakeInferenceServerClient('mnist', response=response)
    client.config.config.max_batch_size = 8
    client.metadata.inputs[0].shape[:] = [-1, 1, 28, 28]
    client.metadata.outputs[0].shape[:] = [-1, 10]
    return triton_api.Model(client, 'mnist'), client


def test_batching_model_accepts_a_batch_and_checks_shape():
    model, client = make_batched_mnist(response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    model.infer([Image.new('L', (28, 28))] * 2)
    assert client.last_request.input_shapes == [[2, 1, 28, 28]]


def test_batching_model_enforces_its_batch_capacity():
    model, _ = make_batched_mnist(response='gradient')
    model.Input3 = triton_api.ImageInput(layout='NCHW')
    with pytest.raises(ValueError, match='at most 8'):
        model.infer([Image.new('L', (28, 28))] * 9)


def test_batching_model_with_dynamic_dims_accepts_size():
    '''The -1 tolerance must hold on the batching branch of infer() too.'''
    client = FakeInferenceServerClient('mnist', response='gradient')
    client.config.config.max_batch_size = 8
    client.metadata.inputs[0].shape[:] = [-1, 1, -1, -1]
    client.metadata.outputs[0].shape[:] = [-1, 10]
    model = triton_api.Model(client, 'mnist')

    model.Input3 = triton_api.ImageInput(layout='NCHW', size=(28, 28))
    model.Plus214_Output_0 = triton_api.TensorOutput()

    result = model.infer(Image.new('L', (28, 28)))
    assert np.asarray(result.Plus214_Output_0).shape == (1, 10)


def test_explicit_batch_axis_accepts_multiple_images():
    '''A max_batch_size=0 model with its own dynamic leading dim batches.'''
    client = FakeInferenceServerClient('mnist', response='gradient')
    client.metadata.inputs[0].shape[:] = [-1, 1, 28, 28]
    model = triton_api.Model(client, 'mnist')

    model.Input3 = triton_api.ImageInput(layout='NCHW')
    model.Plus214_Output_0 = triton_api.TensorOutput()

    model.infer([Image.new('L', (28, 28))] * 2)
    assert client.last_request.input_shapes == [[2, 1, 28, 28]]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
