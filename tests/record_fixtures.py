#!/usr/bin/env python3
'''
Record real Triton gRPC responses so the test suite can replay them offline.

Run this against a live Triton server that is serving the models named below;
it writes the raw protobuf messages under tests/fixtures/<model>/. Because the
fixtures are the servers's own serialized protobufs, fake_triton.py can hand
back byte-identical responses without a server present.

    python3 tests/record_fixtures.py -u localhost:8001

Re-record only when the models or the Triton version change.
'''

import argparse
import os
import sys

import numpy as np
import tritonclient.grpc
import tritonclient.utils

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

import triton_api  # noqa: E402
# The fixture path/compression convention and index schema live in
# fake_triton; writing through it keeps recorder and replayer in lockstep.
from fake_triton import FIXTURES, write_fixture, write_index  # noqa: E402

# Models to record, and the inputs to record responses for. Inputs are
# described as plain shapes/fills so this script needs no image assets.
RECORDINGS = [
    {
        'model': 'mnist',
        # MNIST is a classifier: record both the raw logits and Triton's own
        # class_count post-processing, which is what ClassificationOutput parses.
        'requests': [
            {'name': 'zeros', 'fill': 0.0, 'class_count': 0},
            {'name': 'gradient', 'fill': None, 'class_count': 0},
            {'name': 'gradient_top3', 'fill': None, 'class_count': 3},
        ],
    },
    {
        'model': 'yolov8n',
        # A detection head. The gradient is a synthetic edge case; bus.jpg is a
        # real photo that produces real, nameable COCO detections, so the tests
        # can assert on actual objects rather than just tensor shapes.
        'requests': [
            {'name': 'gradient', 'fill': None, 'class_count': 0},
            {'name': 'bus', 'image': 'assets/bus.jpg', 'class_count': 0},
        ],
    },
]


def make_input(shape, dtype, request, model_obj, input_name):
    '''
    Build the input array for a recording.

    Either a deterministic synthetic pattern (a constant fill or a ramp), or a
    real image preprocessed by triton_api.ImageInput itself -- the very code
    the tests exercise -- so the recorded response corresponds by construction
    to an input the tests can rebuild.
    '''
    if request.get('image'):
        from PIL import Image

        image_input = triton_api.ImageInput(
            scaling=triton_api.ScalingMode.NORM, layout='NCHW',
            letterbox=True)
        setattr(model_obj, input_name, image_input)
        return image_input.process(Image.open(os.path.join(HERE,
                                                           request['image'])))

    size = int(np.prod(shape))
    if request.get('fill') is None:
        data = (np.arange(size, dtype=np.float64) % 255.0) / 255.0
    else:
        data = np.full(size, request['fill'], dtype=np.float64)
    return data.reshape(shape).astype(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default='localhost:8001')
    args = parser.parse_args()

    triton = tritonclient.grpc.InferenceServerClient(url=args.url)

    for entry in RECORDINGS:
        model = entry['model']

        config = triton.get_model_config(model_name=model, as_json=False)
        metadata = triton.get_model_metadata(model_name=model, as_json=False)

        write_fixture(model, 'config.pb', config.SerializeToString())
        write_fixture(model, 'metadata.pb', metadata.SerializeToString())

        # A live Model, so image inputs can be preprocessed by the real
        # ImageInput pipeline rather than a re-implementation of it here.
        model_obj = triton_api.Model(triton, model)

        meta_in = metadata.inputs[0]
        shape = [d if d > 0 else 1 for d in meta_in.shape]
        dtype = tritonclient.utils.triton_to_np_dtype(meta_in.datatype)
        print(f'{model}: input {meta_in.name} {list(meta_in.shape)} '
              f'{meta_in.datatype} -> recording {len(entry["requests"])}')

        index = {'input_name': meta_in.name, 'shape': shape,
                 'datatype': meta_in.datatype, 'requests': []}

        for request in entry['requests']:
            array = make_input(shape, dtype, request, model_obj, meta_in.name)

            infer_input = tritonclient.grpc.InferInput(
                meta_in.name, list(array.shape), meta_in.datatype)
            infer_input.set_data_from_numpy(array)

            requested = [
                tritonclient.grpc.InferRequestedOutput(
                    o.name, class_count=request['class_count'])
                for o in metadata.outputs
            ]

            result = triton.infer(model_name=model, inputs=[infer_input],
                                  outputs=requested)

            # result.get_response() is the ModelInferResponse protobuf; storing
            # it verbatim is what makes the replay faithful.
            name = request['name']
            write_fixture(model, f'response_{name}.pb',
                          result.get_response().SerializeToString())

            index['requests'].append({
                'name': name,
                'fill': request.get('fill'),
                'image': request.get('image'),
                'class_count': request['class_count'],
            })
            print(f'  recorded {name} (class_count={request["class_count"]})')

        write_index(model, index)

    print('fixtures written to', FIXTURES)


if __name__ == '__main__':
    main()
