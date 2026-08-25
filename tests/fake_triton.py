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


def _read(path):
    with gzip.open(path, 'rb') as f:
        return f.read()


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

    def input_array(self, index=0):
        '''The numpy data the library attached to an input, as sent.'''
        return self.inputs[index]._raw_data_view() \
            if hasattr(self.inputs[index], '_raw_data_view') \
            else self.inputs[index]._input.contents


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
    '''

    def __init__(self, model, response=None):
        self.model = model
        self.directory = os.path.join(FIXTURES, model)
        if not os.path.isdir(self.directory):
            available = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(FIXTURES, '*')))
            raise ValueError(
                f'No fixtures for model {model!r}; recorded: {available}')

        with open(os.path.join(self.directory, 'index.json')) as f:
            self.index = json.load(f)

        self.response = response or self.index['requests'][0]['name']

        # Every request the library made, in order, for tests to assert on.
        self.requests = []

    # -- the InferenceServerClient interface the library uses ---------------

    def get_model_config(self, model_name, model_version='', as_json=False):
        response = service_pb2.ModelConfigResponse()
        response.ParseFromString(_read(os.path.join(self.directory, 'config.pb.gz')))
        return response

    def get_model_metadata(self, model_name, model_version='', as_json=False):
        response = service_pb2.ModelMetadataResponse()
        response.ParseFromString(
            _read(os.path.join(self.directory, 'metadata.pb.gz')))
        return response

    def infer(self, model_name, inputs, model_version='', outputs=None,
              **kwargs):
        self.requests.append(RecordedRequest(model_name, model_version,
                                             list(inputs),
                                             list(outputs or [])))

        path = os.path.join(self.directory, f'response_{self.response}.pb.gz')
        if not os.path.exists(path):
            raise ValueError(f'No recorded response named {self.response!r}')

        message = service_pb2.ModelInferResponse()
        message.ParseFromString(_read(path))
        return tritonclient.grpc.InferResult(message)

    # -- convenience for tests ---------------------------------------------

    @property
    def last_request(self):
        assert self.requests, 'no inference request was made'
        return self.requests[-1]
