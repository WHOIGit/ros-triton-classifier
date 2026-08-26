#!/usr/bin/env python3
'''
This file implements a wrapper API over the Triton Inference Server client API.

The wrapper API makes it easier to work with models that require pre- or post-
processing of their inptus and outputs, like image classification models:

    model = Model(triton, 'feline_breed')
    model.input = ImageInput(scaling=ScalingMode.INCEPTION)
    model.output = ClassificationOutput(classes=1)
    
    r = model.infer(Image.open('maeby.jpg'))
    print(result.output[0].score, result.output.class_name) 
'''

import enum
import functools
import warnings

from typing import Any, List, NamedTuple, Union, cast

import numpy as np
import tritonclient.grpc

from PIL import Image
from tritonclient.grpc import model_config_pb2, service_pb2
from tritonclient.utils import triton_to_np_dtype


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    '''
    Convert a *complete* vector of logits into probabilities.

    Triton does not apply an activation function to model outputs. Apply this to
    the full output tensor (bind the output as a TensorOutput); applying it to
    the truncated top-N scores from ClassificationOutput gives wrong answers.
    '''
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def model_dtype_to_np(model_dtype: str) -> np.dtype:
    # tritonclient ships the full Triton -> numpy mapping (including UINT32,
    # UINT64, and BF16, which a hand-rolled table here used to omit).
    np_dtype = triton_to_np_dtype(model_dtype)
    if np_dtype is None:
        raise ValueError(f'Unsupported model datatype {model_dtype!r}')
    return np.dtype(np_dtype)


def shape_matches(actual, expected) -> bool:
    '''
    Compare a concrete tensor shape against a model-declared one, where -1
    marks a dimension the model leaves dynamic (any size satisfies it).
    '''
    actual, expected = list(actual), list(expected)
    return len(actual) == len(expected) and \
        all(e == -1 or a == e for a, e in zip(actual, expected))


class ScalingMode(enum.Enum):
    '''
    Selector for a scaling function to be applied to pixels of an image.
    For example, Inception models expect pixel values in the range -1..1.

    This is unrelated to rescaling the image to new dimensions.
    '''
    NONE = 0
    INCEPTION = 1
    VGG = 2
    NORM = 3  # scale pixel values to the range 0..1 (divide by 255), e.g. YOLO


class Classification(NamedTuple):
    '''
    A single possible classification result for an input. Multiple of these
    Classification objects may be returned.

    Note that the score is a "raw" value, not a percentage; apply softmax or
    a similar function for that.
    '''
    score: float
    class_id: int
    class_name: str


class Detection(NamedTuple):
    '''
    A single detected object: its class, confidence, and bounding box in
    corner form, in pixels relative to the model's input size.
    '''
    score: float
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float


class InferenceResult:
    '''
    Holds the results of an inference call.

    Output values are accessed as attributes of this object.
    '''
    pass


class ModelInput:
    def __init__(self) -> None:
        self.model: Model = None  # type: ignore
        self.name: str = None  # type: ignore
        self.config: model_config_pb2.ModelInput = None  # type: ignore
        self.metadata: service_pb2.ModelMetadataResponse.TensorMetadata = \
            None  # type: ignore

    def bind(self, model: 'Model', name: str):
        '''
        Bind this ModelInput instance to a particular input of a particular
        model.

        This initializes some attributes (self.config, self.metadata) which can
        be accessed later during processing of an input value.
        '''
        assert self.model is None
        self.model, self.name = model, name

        try:
            self.config = next(x for x in model.config.input
                               if x.name == name)
            self.metadata = next(x for x in model.metadata.inputs
                                 if x.name == name)
        except StopIteration as exc:
            raise ValueError(f'No model output named "{name}"') from exc

    def process(self, value: Any) -> np.ndarray:
        '''
        Process the input value into a "raw" ndarray ready for inference.

        It is the responsibility of this function to make sure that the returned
        array is the correct shape. This means that a single input must be 
        reshaped for a model that expects batched input.

        This method is called automatically by Model.infer().
        '''
        raise NotImplementedError()


class TensorInput(ModelInput):
    def process(self, value: np.ndarray) -> np.ndarray:
        # If we received a single input and the tensor carries a batch axis --
        # either implicitly (max_batch_size > 0) or an explicit dynamic leading
        # dimension -- reshape into a batch of one.
        shape = self.metadata.shape
        if len(shape) == value.ndim + 1 and \
                (self.model.can_batch or shape[0] == -1):
            return value.reshape([1] + list(value.shape))

        # Otherwise, pass through unmodified
        return value


class ImageInput(ModelInput):
    # `letterbox` is keyword-only so the pre-existing positional order
    # (scaling, layout, size) stays stable for callers.
    def __init__(self, scaling: ScalingMode = ScalingMode.NONE,
                 layout: str = None,  # type: ignore[assignment]
                 size: tuple = None,  # type: ignore[assignment]
                 *, letterbox: bool = False):
        super().__init__()
        self.scaling = scaling
        # Preserve aspect ratio by scaling to fit and padding the remainder,
        # instead of stretching the image to the model's exact input size.
        # Detection models (YOLO and friends) are trained this way, and feeding
        # them a stretched image silently shifts every predicted box.
        self.letterbox = letterbox
        # Channel layout, 'NCHW' or 'NHWC'. If None it is taken from the model
        # config's `format` field; pass it explicitly for models whose config
        # does not declare a format (common for ONNX detectors, where `format`
        # cannot be set because the input tensor keeps an explicit batch dim).
        self.layout = layout
        # Explicit (width, height) for models that declare dynamic spatial
        # dimensions, where the shape metadata alone cannot tell us the size.
        self.size = size
        self.channels = self.width = self.height = 0
        self._rank = 0  # rank of the model's input tensor (3 or 4)

    def bind(self, model: 'Model', name: str):
        super().bind(model, name)

        shape = list(self.metadata.shape)
        self._rank = len(shape)
        if self._rank not in (3, 4):
            raise ValueError(
                f'ImageInput expects a rank-3 or rank-4 input, got {shape}')

        # Determine the channel layout: an explicit override wins, otherwise
        # fall back to the model config's declared format.
        layout = self.layout
        if layout is None:
            fmt = self.config.format
            if fmt == model_config_pb2.ModelInput.Format.FORMAT_NCHW:
                layout = 'NCHW'
            elif fmt == model_config_pb2.ModelInput.Format.FORMAT_NHWC:
                layout = 'NHWC'
            else:
                raise ValueError(
                    'Model config does not declare an input format; pass '
                    "layout='NCHW' or 'NHWC' to ImageInput()")
        if layout not in ('NCHW', 'NHWC'):
            raise ValueError(f'Unknown layout {layout!r}')
        self.layout = layout

        # Channels/height/width are the trailing three dims; any leading batch
        # dimension (fixed like [1, ...] or dynamic like [-1, ...]) is ignored.
        if layout == 'NCHW':
            self.channels, self.height, self.width = shape[-3], shape[-2], shape[-1]
        else:  # NHWC
            self.height, self.width, self.channels = shape[-3], shape[-2], shape[-1]

        # Triton reports -1 for any dimension the model leaves dynamic, which
        # is not a usable image size. Fall back to an explicit size= for those
        # dimensions only; a size= that contradicts a fixed declared dimension
        # is a configuration error, not an override.
        if self.size is not None:
            want_w, want_h = self.size
            if (self.width > 0 and self.width != want_w) or \
                    (self.height > 0 and self.height != want_h):
                raise ValueError(
                    f'size={self.size} contradicts the model-declared input '
                    f'size ({self.width}, {self.height})')
            if self.width < 1:
                self.width = want_w
            if self.height < 1:
                self.height = want_h
        if self.width < 1 or self.height < 1:
            raise ValueError(
                f'Model declares dynamic spatial dimensions {shape}; pass '
                'size=(width, height) to ImageInput() to choose the input size')
        if self.channels < 1:
            raise ValueError(
                f'Model declares a dynamic channel count in {shape}; '
                'ImageInput needs a fixed 1- or 3-channel input')

        # How many images one request may carry: the model either batches via
        # max_batch_size (Triton adds the axis itself), or declares an explicit
        # leading batch dimension of its own (-1 for unbounded, or a fixed
        # count). None means unbounded.
        if model.can_batch:
            self._capacity = model.max_batch_size
        elif self._rank == 4:
            self._capacity = shape[0] if shape[0] > 0 else None
        else:
            self._capacity = 1

        # Resolve the tensor dtype once, and catch a scaling mode bound to an
        # integer tensor here rather than on the first frame: scaling produces
        # fractional values, so it is only meaningful for a float tensor.
        self.dtype = model_dtype_to_np(self.metadata.datatype)
        if self.scaling != ScalingMode.NONE and \
                not np.issubdtype(self.dtype, np.floating):
            raise ValueError(
                f'{self.scaling.name} scaling requires a floating-point input '
                f'tensor, but the model expects {self.metadata.datatype}')

    def _process_one(self, image: Image.Image) -> np.ndarray:
        # Convert the image to the model's expected channel count
        if self.channels == 1:
            image = image.convert('L')
        elif self.channels == 3:
            image = image.convert('RGB')
        else:
            raise ValueError('Expected grayscale or RGB image')

        # Scale the image down to size. Bilinear scaling is fine:
        # https://medium.com/neuronio/how-to-deal-with-image-resizing-in-deep-learning-e5177fad7d89
        if self.letterbox:
            image = self._letterbox(image)
        else:
            image = image.resize((self.width, self.height), Image.BILINEAR)

        # Convert the image to an ndarray, casting straight into the model's
        # dtype (resolved once at bind time) in a single pass.
        array = np.array(image, dtype=self.dtype)

        # If the image is grayscale, add a channel axis (HW -> HWC)
        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        # Apply optional pixel scaling.
        if self.scaling == ScalingMode.NONE:
            pass
        elif self.scaling == ScalingMode.NORM:
            array /= 255.0
        elif self.scaling == ScalingMode.INCEPTION:
            array /= 127.5
            array -= 1
        else:
            raise NotImplementedError('Scaling mode is not implemented yet')

        return array

    def _letterbox_geometry(self, source_size: tuple) -> tuple:
        '''
        The single definition of where a letterboxed source image lands in
        the input frame: the resized (width, height) and the top-left pad
        offsets. Both _letterbox and source_mapping derive from this, so the
        forward transform and its inverse cannot drift apart.
        '''
        source_width, source_height = source_size
        scale = min(self.width / source_width, self.height / source_height)
        new_width = max(1, round(source_width * scale))
        new_height = max(1, round(source_height * scale))
        return (new_width, new_height,
                (self.width - new_width) // 2, (self.height - new_height) // 2)

    def source_mapping(self, source_size: tuple) -> tuple:
        '''
        Describe how a source image of ``source_size`` = (width, height) is
        placed into the model's input frame.

        Returns ``(scale_x, scale_y, pad_x, pad_y)`` such that

            input_x = source_x * scale_x + pad_x
            input_y = source_y * scale_y + pad_y

        Detection coordinates come back in the model's input frame, so this is
        what you need to put them back on the original image.
        '''
        source_width, source_height = source_size
        if not self.letterbox:
            return (self.width / source_width, self.height / source_height,
                    0.0, 0.0)

        # The effective per-axis scale is the ratio of the *rounded* resize
        # to the source -- not the ideal min() ratio -- otherwise inverting
        # the mapping drifts by up to a pixel in the input frame (and several
        # pixels in a large source image).
        new_width, new_height, pad_x, pad_y = \
            self._letterbox_geometry(source_size)
        return (new_width / source_width, new_height / source_height,
                pad_x, pad_y)

    def to_source_box(self, box: tuple, source_size: tuple) -> tuple:
        '''
        Map an (x1, y1, x2, y2) box from the model's input frame back onto a
        source image of ``source_size``, clipped to that image.
        '''
        scale_x, scale_y, pad_x, pad_y = self.source_mapping(source_size)
        x1, y1, x2, y2 = box
        source_width, source_height = source_size
        return (
            min(max((x1 - pad_x) / scale_x, 0.0), source_width),
            min(max((y1 - pad_y) / scale_y, 0.0), source_height),
            min(max((x2 - pad_x) / scale_x, 0.0), source_width),
            min(max((y2 - pad_y) / scale_y, 0.0), source_height),
        )

    # The 114-gray fill matches the Ultralytics letterbox convention.
    LETTERBOX_FILL = 114

    def _letterbox(self, image: Image.Image) -> Image.Image:
        '''
        Resize preserving aspect ratio, centered on a constant-filled canvas of
        the model's input size. This matches how detection models are trained.
        '''
        new_width, new_height, pad_x, pad_y = \
            self._letterbox_geometry((image.width, image.height))
        resized = image.resize((new_width, new_height), Image.BILINEAR)

        background = (self.LETTERBOX_FILL,) * len(image.getbands())
        canvas = Image.new(image.mode, (self.width, self.height), background)
        canvas.paste(resized, (pad_x, pad_y))
        return canvas

    def process(self, value: Union[Image.Image, List[Image.Image]]) \
            -> np.ndarray:
        
        # Temporarily convert value to a list, even for non-batched inputs, for
        # ease of processing.
        if not isinstance(value, list):
            value = [value]
        value = cast(List[Image.Image], value)

        if self._capacity is not None and len(value) > self._capacity:
            raise ValueError(
                f'Model accepts at most {self._capacity} image(s) per request')

        # Process all of the images into a batch in NHWC format
        processed = np.stack([self._process_one(image) for image in value])

        # If the model expects channels-first, re-arrange NHWC -> NCHW
        if self.layout == 'NCHW':
            processed = np.transpose(processed, (0, 3, 1, 2))

        # Keep the leading batch axis only if the model's input tensor has one
        # (rank 4). A rank-3 input wants a single [C,H,W] / [H,W,C] array.
        if self._rank == 3:
            return processed[0]
        return processed


class ModelOutput:
    def __init__(self) -> None:
        self.model: Model = None  # type: ignore
        self.name: str = None  # type: ignore
        self.config: model_config_pb2.ModelOutput = None  # type: ignore
        self.metadata: service_pb2.ModelMetadataResponse.TensorMetadata = \
            None  # type: ignore

    def bind(self, model: 'Model', name: str):
        '''
        Bind this ModelOutput instance to a particular input of a particular
        model.

        This initializes some attributes (self.config, self.metadata) which can
        be accessed later during processing of an output value.
        '''
        assert self.model is None
        self.model, self.name = model, name

        try:
            self.config = next(x for x in model.config.output
                               if x.name == name)
            self.metadata = next(x for x in model.metadata.outputs
                                 if x.name == name)
        except StopIteration as exc:
            raise ValueError(f'No model output named "{name}"') from exc

    def process(self, value: np.ndarray) -> Any:
        '''
        Process the "raw" ndarray output from the inference result and
        potentially convert it to another data type.

        This method is called automatically by Model.infer().
        '''
        raise NotImplementedError()


class TensorOutput(ModelOutput):
    def process(self, value: np.ndarray) -> np.ndarray:
        return value


class ClassificationOutput(ModelOutput):
    '''
    Parses Triton's built-in classification post-processing (`class_count`),
    which returns only the top-N entries as score:class_id:class_name strings.

    NOTE: because the server truncates to the N classes requested, the scores
    this returns are an arbitrary subset of the model's logits. A softmax over
    them normalizes against that subset, not the full class set, and so reports
    confidences that are too high (see issue #5). To obtain real probabilities,
    bind the output as a TensorOutput to get the complete logit vector and call
    softmax() on it client-side.
    '''

    def __init__(self, classes: int = 1):
        super().__init__()
        if classes < 1:
            raise ValueError('Must request at least one class')
        self.classes = classes

    def _parse_classification(self, c: bytes) -> Classification:
        '''
        Parse the score:class_id:class_name format that we receive from Triton
        into a Classification object.

        The trailing name is only present when the model config names a
        label_filename; without one Triton sends just score:class_id.
        '''
        fields = c.decode().split(':', maxsplit=2)
        if len(fields) == 2:
            # Tolerated, but worth one loud note: without label_filename in the
            # model config every published class_name will be empty.
            warnings.warn(
                'Triton returned classifications without class names; the '
                'model config likely does not set label_filename',
                stacklevel=2)
            (score, class_id), class_name = fields, ''
        elif len(fields) == 3:
            score, class_id, class_name = fields
        else:
            raise ValueError(f'Malformed classification from Triton: {c!r}')

        return Classification(
            score=float(score),
            class_id=int(class_id),
            class_name=class_name,
        )

    def process(self, value: np.ndarray) \
            -> Union[List[List[Classification]], List[Classification]]:

        if value.ndim == 2:  # batched
            return [
                [self._parse_classification(b) for b in a]
                for a in value
            ]
        elif value.ndim == 1:  # single fire
            return [self._parse_classification(b) for b in value]
        else:
            raise ValueError('Expected only 1 or 2 dimensions')


class DetectionOutput(ModelOutput):
    '''
    Decodes the raw output of a YOLO-style detection head into Detection
    objects: it splits boxes from class scores, drops low-confidence anchors,
    and applies per-class non-maximum suppression.

    Expects a tensor shaped (4 + num_classes, num_anchors) -- optionally with a
    leading batch axis -- where the first four rows are the box center x, center
    y, width, and height in input-image pixels, as exported by Ultralytics
    YOLOv8/v11. YOLOv5/v7-style exports that carry an extra objectness row
    (4 + 1 + num_classes) are NOT understood; pass `labels` so the class count
    can be checked, which turns that silent mis-decode into an error and also
    resolves the tensor orientation authoritatively. Box coordinates are
    relative to the model's input size; if the image was letterboxed, map them
    back yourself using the same scale/padding.
    '''

    # Upper bound on candidates entering NMS, like Ultralytics' max_det: keeps
    # the greedy loop bounded on cluttered frames.
    MAX_CANDIDATES = 300

    def __init__(self, confidence: float = 0.25, iou: float = 0.45,
                 labels: List[str] = None):  # type: ignore[assignment]
        super().__init__()
        self.confidence = confidence
        self.iou = iou
        self.labels = labels

    def _class_name(self, class_id: int) -> str:
        if self.labels is not None and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
        '''Greedy non-maximum suppression over xyxy boxes; returns kept indices.'''
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            best = order[0]
            keep.append(int(best))
            rest = order[1:]

            # Intersection of the best box with every remaining box
            ix1 = np.maximum(x1[best], x1[rest])
            iy1 = np.maximum(y1[best], y1[rest])
            ix2 = np.minimum(x2[best], x2[rest])
            iy2 = np.minimum(y2[best], y2[rest])
            overlap = (np.clip(ix2 - ix1, 0, None) *
                       np.clip(iy2 - iy1, 0, None))

            union = areas[best] + areas[rest] - overlap
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(union > 0, overlap / union, 0)
            order = rest[ratio <= iou]

        return keep

    def _oriented(self, predictions: np.ndarray) -> np.ndarray:
        '''
        Return the output as (4 + num_classes, num_anchors). Exporters disagree
        about the axis order; with `labels` the class count identifies the
        right axis authoritatively, otherwise fall back to the heuristic that
        anchors vastly outnumber the 4+nc rows.
        '''
        r, c = predictions.shape

        if self.labels is not None:
            rows = 4 + len(self.labels)
            if (r == rows) != (c == rows):
                return predictions if r == rows else predictions.T
            if r != rows and c != rows:
                raise ValueError(
                    f'Neither axis of the {predictions.shape} output matches '
                    f'4 + {len(self.labels)} labels; a YOLOv5/v7-style export '
                    'with an objectness row (4+1+nc) is not supported')
            # Both axes match (square): fall through to the shape heuristic,
            # which cannot decide either.

        if r == c:
            raise ValueError(
                f'Cannot infer the orientation of a square '
                f'{predictions.shape} detection output; pass labels= so the '
                'class count disambiguates it')
        return predictions.T if r > c else predictions

    def _process_one(self, predictions: np.ndarray) -> List[Detection]:
        predictions = self._oriented(predictions)
        if predictions.shape[0] <= 4:
            raise ValueError(
                f'Expected a (4 + num_classes, num_anchors) tensor, got '
                f'{predictions.shape}: a box-only output has no class scores')

        boxes, scores = predictions[:4].T, predictions[4:]

        # Reduce over the class axis first and filter before the (expensive)
        # argmax, which then only runs over the surviving anchors.
        confidences = scores.max(axis=0)
        selected = np.flatnonzero(confidences >= self.confidence)
        if selected.size > self.MAX_CANDIDATES:
            top = np.argpartition(confidences[selected],
                                  -self.MAX_CANDIDATES)[-self.MAX_CANDIDATES:]
            selected = selected[top]

        boxes = boxes[selected]
        confidences = confidences[selected]
        class_ids = np.argmax(scores[:, selected], axis=0)

        if selected.size == 0:
            return []

        # Convert center-form xywh to corner-form xyxy for NMS, and clip to the
        # input bounds -- the head can predict boxes that run off the edge. The
        # bounds come from whichever input is a bound ImageInput; when nothing
        # exposes a size (e.g. a TensorInput fed preprocessed arrays) we cannot
        # clip, and say so once rather than silently skipping.
        centers, half = boxes[:, :2], boxes[:, 2:4] * 0.5
        corners = np.concatenate([centers - half, centers + half], axis=1)
        image_input = next(
            (obj for m in self.model.metadata.inputs
             if isinstance(obj := getattr(self.model, m.name), ImageInput)),
            None)
        if image_input is not None:
            np.clip(corners[:, 0::2], 0, image_input.width,
                    out=corners[:, 0::2])
            np.clip(corners[:, 1::2], 0, image_input.height,
                    out=corners[:, 1::2])
        else:
            warnings.warn('No ImageInput is bound, so detection boxes are not '
                          'clipped to the input bounds', stacklevel=2)

        # Per-class suppression via the offset trick: shift each class's boxes
        # into a disjoint coordinate region so a single NMS pass can never
        # suppress across classes. Any offset wider than the coordinate span
        # works. _nms returns indices best-first, so the result is sorted.
        offset = float(corners.max() - min(corners.min(), 0.0)) + 1.0
        shifted = corners + (class_ids * offset)[:, None]

        detections = []
        for index in self._nms(shifted, confidences, self.iou):
            x1, y1, x2, y2 = corners[index]
            class_id = int(class_ids[index])
            detections.append(Detection(
                score=float(confidences[index]),
                class_id=class_id,
                class_name=self._class_name(class_id),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return detections

    def process(self, value: np.ndarray) \
            -> Union[List[List[Detection]], List[Detection]]:

        # Only a batching model's leading axis is a batch; for a non-batching
        # model a leading 1 belongs to the model's own output shape.
        if self.model.can_batch:
            if value.ndim != 3:
                raise ValueError('Expected 3 dimensions for a batched output')
            return [self._process_one(a) for a in value]

        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError('Expected a 2-dimensional detection output')
        return self._process_one(value)


class Model:
    def __init__(self,
                 triton: tritonclient.grpc.InferenceServerClient,
                 name: str,
                 version: str = ''):
        self.triton = triton
        self.name, self.version = name, version

        # Create default input and output fields
        self._inputs = set()
        for input in self.metadata.inputs:
            assert not hasattr(self, input.name)
            self._inputs.add(input.name)
            setattr(self, input.name, TensorInput())

        self._outputs = set()
        for output in self.metadata.outputs:
            assert not hasattr(self, output.name)
            self._outputs.add(output.name)
            setattr(self, output.name, TensorOutput())

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        # When the user assigns an input or output field, invoke .bind() on that
        # object to associate this model with it.
        if name in getattr(self, '_inputs', set()):
            assert isinstance(value, ModelInput)
            value.bind(self, name)
        elif name in getattr(self, '_outputs', set()) and value is not None:
            assert isinstance(value, ModelOutput)
            value.bind(self, name)

    @functools.cached_property
    def config(self) -> model_config_pb2.ModelConfig:
        '''
        Get the configuration for a given model. This is loaded from the model's
        config.pbtxt file.

        https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
        https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_model_configuration.html#grpc
        '''
        return self.triton.get_model_config(
            model_name=self.name,
            model_version=self.version,
        ).config

    @functools.cached_property
    def metadata(self) -> service_pb2.ModelMetadataResponse:
        '''
        Get metadata for a given model, which includes information about input
        and output tensors.

        Not really documented, but see:
        https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto
        '''
        return self.triton.get_model_metadata(
            model_name=self.name,
            model_version=self.version,
        )

    @property
    def max_batch_size(self) -> int:
        return self.config.max_batch_size

    @property
    def can_batch(self) -> bool:
        return self.max_batch_size > 0

    def infer(self, *args, **kwargs):
        '''
        Submits input values to the inference server.

        If the model accepts batched inputs, and only a single input is
        provided, it will be mutated into a batch of one.
        '''

        if (args and kwargs) or (not args and not kwargs):
            raise ValueError('Provide all inputs as either positional or '
                             'keyword arguments')

        # Convert positional arguments to keyword arguments, using the
        # corresponding input's name from the model metadata.
        if args:
            kwargs = {m.name: a for a, m in zip(args, self.metadata.inputs)}
            del args

        # Check that we received values for all inputs
        expected = set(m.name for m in self.metadata.inputs)
        if kwargs.keys() != expected:
            raise ValueError('Expected values for these inputs: ' +
                             ', '.join(expected))

        # Process each input to an ndarray using the ModelInput subclass. For
        # example, an ImageInput would convert an image into an array.
        #
        # At the end of this loop we'll have a list of InferInput objects ready
        # for the RPC call.
        req_inputs: List[tritonclient.grpc.InferInput] = []

        for key, value in kwargs.items():
            inputobj = getattr(self, key)
            assert isinstance(inputobj, ModelInput)
            result = kwargs[key] = inputobj.process(value)
            assert isinstance(result, np.ndarray)

            # After processing, check that each ndarray's shape matches the
            # model's declared shape (shape_matches treats -1 as a wildcard).
            # A real raise, not an assert: this must survive `python -O`.
            if self.can_batch:
                if not shape_matches(result.shape[1:],
                                     inputobj.metadata.shape[1:]):
                    raise ValueError(
                        f'processed input shape {list(result.shape)} does not '
                        f'match model shape '
                        f'[batch, {", ".join(map(str, inputobj.metadata.shape[1:]))}]')
                if result.shape[0] > self.max_batch_size:
                    raise ValueError('Too many inputs in batch')
            else:
                if not shape_matches(result.shape, inputobj.metadata.shape):
                    raise ValueError(
                        f'processed input shape {list(result.shape)} does not '
                        f'match model shape {list(inputobj.metadata.shape)}')

            # Create the InferInput object for this input
            req_inputs.append(tritonclient.grpc.InferInput(
                name=inputobj.name,
                datatype=inputobj.metadata.datatype,
                shape=result.shape,
            ))
            req_inputs[-1].set_data_from_numpy(result)

        # Build a list of InferRequestedOutput to request each of the configured
        # outputs.
        req_outputs: List[tritonclient.grpc.InferRequestedOutput] = []
        for output in self.metadata.outputs:
            outputobj = getattr(self, output.name)
            if outputobj is None:
                continue
            assert isinstance(outputobj, ModelOutput)

            req_outputs.append(tritonclient.grpc.InferRequestedOutput(
                name=outputobj.name,
                class_count=getattr(outputobj, 'classes', 0)  # hacky
            ))

        # Submit the request to the inference server!
        response = self.triton.infer(
            model_name=self.name,
            model_version=self.version,
            inputs=req_inputs,
            outputs=req_outputs,
        )

        # Postprocess the values returned from the server and return them as
        # attributes on an InferenceResult object.
        result = InferenceResult()
        for output in req_outputs:
            outputobj = getattr(self, output.name())
            assert isinstance(outputobj, ModelOutput)
            value = cast(np.ndarray, response.as_numpy(output.name()))
            setattr(result, output.name(), outputobj.process(value))

        return result

def initialize_model(url, model_name, verbose=False, model_version=''):
    # Create a Triton client using the gRPC transport
    triton = tritonclient.grpc.InferenceServerClient(
        url=url,
        verbose=verbose
    )

    # Create the model
    return Model(
        triton,
        model_name,
        model_version,
    )

def main():
    import argparse
    import pprint
    import time

    parser = argparse.ArgumentParser(
        description='Submit image(s) to a Triton model and print the results.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--model-name', required=True)
    parser.add_argument('-x', '--model-version', default='')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='images to submit per inference request; the '
                             "model's max_batch_size is the upper limit")
    # VGG is deliberately absent: ScalingMode.VGG has no implementation, and
    # offering it here only defers the failure to inference time.
    parser.add_argument('-t', '--image-transform',
                        choices=['NONE', 'NORM', 'INCEPTION'],
                        default='NONE')
    parser.add_argument('-l', '--layout', choices=['NCHW', 'NHWC'], default=None,
                        help="channel layout when the model config omits `format`")
    parser.add_argument('--letterbox', action='store_true',
                        help='preserve aspect ratio by padding instead of '
                             'stretching (detection models expect this)')
    parser.add_argument('-s', '--size', type=int, nargs=2, default=None,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='input size for models that declare dynamic '
                             'spatial dimensions')
    parser.add_argument('-u', '--url', default='localhost:8001')

    # How to interpret the output tensor. These are mutually exclusive so that
    # e.g. `-c 5 --raw` is rejected rather than silently ignoring -c.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-c', '--classes', type=int, default=None,
                      help='parse the top N classifications (the default, N=3)')
    mode.add_argument('--detect', action='store_true',
                      help='decode YOLO-style detections (boxes + NMS)')
    mode.add_argument('--raw', action='store_true',
                      help='return the raw output tensor without parsing')

    parser.add_argument('images', nargs='+')
    args = parser.parse_args()

    # Detection models are trained on letterboxed input; a stretched image
    # silently shifts every predicted box, so --detect implies --letterbox.
    if args.detect:
        args.letterbox = True

    model = initialize_model(args.url, args.model_name, args.verbose,
                             args.model_version)

    # Bind the first input as an image and the first output for our result,
    # by their actual tensor names (not every model calls them input/output).
    in_name = model.metadata.inputs[0].name
    out_name = model.metadata.outputs[0].name

    setattr(model, in_name, ImageInput(
        scaling=ScalingMode[args.image_transform], layout=args.layout,
        letterbox=args.letterbox,
        size=tuple(args.size) if args.size else None))
    if args.raw:
        setattr(model, out_name, TensorOutput())
    elif args.detect:
        setattr(model, out_name, DetectionOutput())
    else:
        setattr(model, out_name, ClassificationOutput(
            classes=args.classes if args.classes is not None else 3))

    images = [Image.open(path) for path in args.images]

    # Submit in batches so a batch-capable model can amortize the round trip.
    batch_size = max(1, args.batch_size)
    if batch_size > 1 and not model.can_batch:
        parser.error(f'model {model.name} does not accept batched input')
    if model.can_batch:
        batch_size = min(batch_size, model.max_batch_size)

    start = time.perf_counter()
    for offset in range(0, len(images), batch_size):
        batch = images[offset:offset + batch_size]

        # A batching model always wants a list, even a single-element one.
        result = model.infer(batch if model.can_batch else batch[0])
        value = getattr(result, out_name)
        if args.raw:
            arr = np.asarray(value)
            # min/max only exist for a non-empty numeric tensor (a detector
            # may legitimately return zero rows, or a BYTES output).
            stats = ''
            if arr.size and np.issubdtype(arr.dtype, np.number):
                stats = f' min={float(arr.min()):.4f} max={float(arr.max()):.4f}'
            print(f'{out_name}: shape={arr.shape} dtype={arr.dtype}{stats}')
        else:
            pprint.pprint(value)
    stop = time.perf_counter()

    n = len(images)
    print(f'Processed {n} image(s) in {(stop - start):0.3f} s '
          f'({(stop - start) / n:0.3f} s per image)')


if __name__ == '__main__':
    main()
