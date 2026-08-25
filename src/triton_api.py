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

from typing import Any, List, NamedTuple, Union, cast

import numpy as np
import tritonclient.grpc

from PIL import Image
from tritonclient.grpc import model_config_pb2, service_pb2


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


def model_dtype_to_np(model_dtype: str) -> object:
    return {
        'BOOL':   bool,
        'INT8':   np.int8,
        'INT16':  np.int16,
        'INT32':  np.int32,
        'INT64':  np.int64,
        'UINT8':  np.uint8,
        'UINT16': np.uint16,
        'FP16':   np.float16,
        'FP32':   np.float32,
        'FP64':   np.float64,
        'BYTES':  np.dtype(object),
    }[model_dtype]


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
        # If we received a single input and expected a batch, reshape
        if self.model.can_batch and len(self.metadata.shape) == value.ndim + 1:
            return value.reshape([1] + list(value.shape))

        # Otherwise, pass through unmodified
        return value


class ImageInput(ModelInput):
    def __init__(self, scaling: ScalingMode = ScalingMode.NONE,
                 layout: str = None,  # type: ignore[assignment]
                 size: tuple = None):  # type: ignore[assignment]
        super().__init__()
        self.scaling = scaling
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

        # Triton reports -1 for any dimension the model leaves dynamic, which is
        # not a usable image size. Fall back to an explicit size= if given.
        if self.size is not None:
            self.width, self.height = self.size
        if self.width < 1 or self.height < 1:
            raise ValueError(
                f'Model declares dynamic spatial dimensions {shape}; pass '
                'size=(width, height) to ImageInput() to choose the input size')
        if self.channels < 1:
            raise ValueError(
                f'Model declares a dynamic channel count in {shape}; '
                'ImageInput needs a fixed 1- or 3-channel input')

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
        image = image.resize((self.width, self.height), Image.BILINEAR)

        # Convert the image to a nparray. Scaling produces fractional values, so
        # it is only meaningful for a float input tensor; catch an integer one
        # here rather than letting numpy raise an opaque casting error below.
        dtype = np.dtype(model_dtype_to_np(self.metadata.datatype))
        if self.scaling != ScalingMode.NONE and not np.issubdtype(dtype,
                                                                  np.floating):
            raise ValueError(
                f'{self.scaling.name} scaling requires a floating-point input '
                f'tensor, but the model expects {self.metadata.datatype}')
        array = np.array(image).astype(dtype)

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

    def process(self, value: Union[Image.Image, List[Image.Image]]) \
            -> np.ndarray:
        
        # Temporarily convert value to a list, even for non-batched inputs, for
        # ease of processing.
        if not isinstance(value, list):
            value = [value]
        value = cast(List[Image.Image], value)

        if not self.model.can_batch and len(value) != 1:
            raise ValueError('Input expects exactly one image')

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
        '''
        score, class_id, class_name = c.decode().split(':', maxsplit=2)
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

            # After processing, assert that each ndarray's shape matches the
            # model's expected shape.
            if self.can_batch:
                assert result.shape[1:] == tuple(inputobj.metadata.shape[1:])
                if result.shape[0] > self.max_batch_size:
                    raise ValueError('Too many inputs in batch')
            else:
                # -1 marks a dimension the model leaves dynamic, which any
                # size satisfies.
                actual, expected = \
                    list(result.shape), list(inputobj.metadata.shape)
                assert len(actual) == len(expected) and \
                    all(e == -1 or a == e for a, e in zip(actual, expected)), \
                    (f'processed input shape {actual} does not '
                     f'match model shape {expected}')

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
    parser.add_argument('-c', '--classes', type=int, default=3,
                        help='classes to request (classification mode)')
    parser.add_argument('-t', '--image-transform',
                        choices=['NONE', 'NORM', 'INCEPTION', 'VGG'],
                        default='NONE')
    parser.add_argument('-l', '--layout', choices=['NCHW', 'NHWC'], default=None,
                        help="channel layout when the model config omits `format`")
    parser.add_argument('--raw', action='store_true',
                        help='return the raw output tensor instead of parsing '
                             'classifications (e.g. for detection models)')
    parser.add_argument('-u', '--url', default='localhost:8001')
    parser.add_argument('images', nargs='+')
    args = parser.parse_args()

    model = initialize_model(args.url, args.model_name, args.verbose,
                             args.model_version)

    # Bind the first input as an image and the first output for our result,
    # by their actual tensor names (not every model calls them input/output).
    in_name = model.metadata.inputs[0].name
    out_name = model.metadata.outputs[0].name

    setattr(model, in_name, ImageInput(
        scaling=ScalingMode[args.image_transform], layout=args.layout))
    if args.raw:
        setattr(model, out_name, TensorOutput())
    else:
        setattr(model, out_name, ClassificationOutput(classes=args.classes))

    images = [Image.open(path) for path in args.images]

    start = time.perf_counter()
    for image in images:
        result = model.infer(image)
        value = getattr(result, out_name)
        if args.raw:
            arr = np.asarray(value)
            print(f'{out_name}: shape={arr.shape} dtype={arr.dtype} '
                  f'min={float(arr.min()):.4f} max={float(arr.max()):.4f}')
        else:
            pprint.pprint(value)
    stop = time.perf_counter()

    n = len(images)
    print(f'Processed {n} image(s) in {(stop - start):0.3f} s '
          f'({(stop - start) / n:0.3f} s per image)')


if __name__ == '__main__':
    main()
