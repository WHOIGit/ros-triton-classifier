'''
An offline stand-in for tritonclient.grpc.InferenceServerClient.

It replays responses recorded from a real Triton server by
tests/record_fixtures.py. The fixtures are the server's own serialized
protobufs, so the objects handed back here -- model config, model metadata, and
InferResult -- are the same types, with the same contents, that the real client
would return. That keeps the tests honest about the client library's behaviour
without needing a GPU, a server, or a network.

What it deliberately does NOT simulate: actually running the model. A replayed
response is tied to the input it was recorded with, so tests assert on how the
library builds requests and interprets responses, not on inference itself.
'''

import glob
import gzip
import json
import os

from tritonclient.grpc import service_pb2
import tritonclient.grpc

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

# The fixture naming and compression convention lives HERE, and only here:
# record_fixtures.py writes through these helpers and this module reads
# through them, so the two sides cannot drift apart.


def fixture_path(model, filename):
    '''Path of one gzip-compressed fixture file, e.g. ('mnist', 'config.pb').'''
    return os.path.join(FIXTURES, model, filename + '.gz')


def read_fixture(model, filename):
    with gzip.open(fixture_path(model, filename), 'rb') as f:
        return f.read()


def write_fixture(model, filename, data):
    os.makedirs(os.path.join(FIXTURES, model), exist_ok=True)
    with gzip.open(fixture_path(model, filename), 'wb') as f:
        f.write(data)


def read_index(model):
    with open(os.path.join(FIXTURES, model, 'index.json')) as f:
        return json.load(f)


def write_index(model, index):
    os.makedirs(os.path.join(FIXTURES, model), exist_ok=True)
    with open(os.path.join(FIXTURES, model, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2, sort_keys=True)


class RecordedRequest:
    '''One inference request as the library issued it, captured for assertions.'''

    def __init__(self, model_name, model_version, inputs, outputs):
        self.model_name = model_name
        self.model_version = model_version
        self.inputs = inputs
        self.outputs = outputs

    @property
    def input_shapes(self):
        return [list(i.shape()) for i in self.inputs]

    @property
    def input_datatypes(self):
        return [i.datatype() for i in self.inputs]

    @property
    def input_names(self):
        return [i.name() for i in self.inputs]

    @property
    def output_names(self):
        return [o.name() for o in self.outputs]

    @property
    def output_class_counts(self):
        '''The server-side class_count requested for each output.'''
        # Knowledge of InferRequestedOutput's internals stays inside this
        # module so tests never touch tritonclient privates directly.
        return [
            o._output.parameters['classification'].int64_param
            if 'classification' in o._output.parameters else 0
            for o in self.outputs
        ]


class FakeInferenceServerClient:
    '''
    Drop-in replacement for the gRPC client, backed by recorded fixtures.

    Parameters
    ----------
    model:
        Fixture directory name, e.g. 'mnist' or 'yolov8n'.
    response:
        Which recorded response to replay from infer(); defaults to the first
        one in the fixture index. Change it between calls to replay another.
    model_version:
        The version the library is expected to request; '' accepts only
        version-less requests.

    The config and metadata protobufs are parsed once at construction and
    exposed as `self.config` / `self.metadata`; tests may mutate them (e.g.
    set max_batch_size, or make a dimension dynamic) BEFORE constructing the
    Model, to exercise library paths no recorded fixture reaches.

    infer() validates each request against the recording -- model name and
    version, input tensor name/shape/datatype, and the requested class_count
    -- so a library regression in request construction fails the test that
    triggered it instead of silently replaying an unrelated response.
    '''

    def __init__(self, model, response=None, model_version=''):
        self.model = model
        self.model_version = model_version
        self.directory = os.path.join(FIXTURES, model)
        if not os.path.isdir(self.directory):
            available = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(FIXTURES, '*')))
            raise ValueError(
                f'No fixtures for model {model!r}; recorded: {available}')

        self.index = read_index(model)
        self.response = response or self.index['requests'][0]['name']

        self.config = service_pb2.ModelConfigResponse()
        self.config.ParseFromString(read_fixture(model, 'config.pb'))
        self.metadata = service_pb2.ModelMetadataResponse()
        self.metadata.ParseFromString(read_fixture(model, 'metadata.pb'))

        # Every request the library made, in order, for tests to assert on.
        self.requests = []

    def _check_target(self, what, model_name, model_version):
        if model_name != self.model:
            raise AssertionError(
                f'{what} requested model {model_name!r}, but this fake '
                f'serves {self.model!r}')
        if model_version != self.model_version:
            raise AssertionError(
                f'{what} requested model version {model_version!r}, '
                f'expected {self.model_version!r}')

    # -- the InferenceServerClient interface the library uses ---------------

    def get_model_config(self, model_name, model_version='', as_json=False):
        self._check_target('get_model_config', model_name, model_version)
        return self.config

    def get_model_metadata(self, model_name, model_version='', as_json=False):
        self._check_target('get_model_metadata', model_name, model_version)
        return self.metadata

    def infer(self, model_name, inputs, model_version='', outputs=None,
              **kwargs):
        self._check_target('infer', model_name, model_version)
        request = RecordedRequest(model_name, model_version,
                                  list(inputs), list(outputs or []))
        self.requests.append(request)

        # Validate the request against what the fixture was recorded with.
        # Only the leading batch size may differ from the recorded shape;
        # everything else drifting means the library built a different
        # request than the recorded response answers.
        entry = next((r for r in self.index['requests']
                      if r['name'] == self.response), None)
        if entry is None:
            raise ValueError(f'No recorded response named {self.response!r}')
        expected_name = self.index['input_name']
        expected_shape = list(self.index['shape'])
        expected_dtype = self.index['datatype']
        for infer_input in request.inputs:
            if infer_input.name() != expected_name:
                raise AssertionError(
                    f'infer() sent input {infer_input.name()!r}, recorded '
                    f'input is {expected_name!r}')
            if infer_input.datatype() != expected_dtype:
                raise AssertionError(
                    f'infer() sent datatype {infer_input.datatype()!r}, '
                    f'recorded datatype is {expected_dtype!r}')
            sent = list(infer_input.shape())
            if sent[1:] != expected_shape[1:] or len(sent) != len(expected_shape):
                raise AssertionError(
                    f'infer() sent shape {sent}, recorded shape is '
                    f'{expected_shape}')
        for class_count in request.output_class_counts:
            if class_count != entry['class_count']:
                raise AssertionError(
                    f'infer() requested class_count={class_count}, but '
                    f'{self.response!r} was recorded with '
                    f'class_count={entry["class_count"]}')

        message = service_pb2.ModelInferResponse()
        message.ParseFromString(
            read_fixture(self.model, f'response_{self.response}.pb'))
        return tritonclient.grpc.InferResult(message)

    # -- convenience for tests ---------------------------------------------

    @property
    def last_request(self):
        assert self.requests, 'no inference request was made'
        return self.requests[-1]
