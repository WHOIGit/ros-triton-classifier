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
import json
import os

import numpy as np
import tritonclient.grpc
import tritonclient.utils

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'fixtures')

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


def make_input(shape, dtype, request):
    '''
    Build the input array for a recording.

    Either a deterministic synthetic pattern (a constant fill or a ramp), or a
    real image letterboxed and normalized exactly the way ImageInput would, so
    the recorded response corresponds to an input the tests can rebuild.
    '''
    if request.get('image'):
        return letterbox_image(os.path.join(HERE, request['image']),
                               shape, dtype)

    size = int(np.prod(shape))
    if request.get('fill') is None:
        data = (np.arange(size, dtype=np.float64) % 255.0) / 255.0
    else:
        data = np.full(size, request['fill'], dtype=np.float64)
    return data.reshape(shape).astype(dtype)


def letterbox_image(path, shape, dtype):
    '''Load an image and letterbox it into an NCHW tensor of `shape`, /255.'''
    from PIL import Image

    channels, height, width = shape[-3], shape[-2], shape[-1]
    image = Image.open(path).convert('RGB' if channels == 3 else 'L')

    scale = min(width / image.width, height / image.height)
    new_size = (max(1, round(image.width * scale)),
                max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.BILINEAR)

    fill = 114
    background = fill if image.mode == 'L' else (fill,) * channels
    canvas = Image.new(image.mode, (width, height), background)
    canvas.paste(resized, ((width - new_size[0]) // 2,
                           (height - new_size[1]) // 2))

    array = np.array(canvas).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    array = np.transpose(array, (2, 0, 1))          # HWC -> CHW
    return array.reshape(shape).astype(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default='localhost:8001')
    args = parser.parse_args()

    triton = tritonclient.grpc.InferenceServerClient(url=args.url)

    for entry in RECORDINGS:
        model = entry['model']
        out_dir = os.path.join(FIXTURES, model)
        os.makedirs(out_dir, exist_ok=True)

        config = triton.get_model_config(model_name=model, as_json=False)
        metadata = triton.get_model_metadata(model_name=model, as_json=False)

        with open(os.path.join(out_dir, 'config.pb'), 'wb') as f:
            f.write(config.SerializeToString())
        with open(os.path.join(out_dir, 'metadata.pb'), 'wb') as f:
            f.write(metadata.SerializeToString())

        meta_in = metadata.inputs[0]
        shape = [d if d > 0 else 1 for d in meta_in.shape]
        dtype = tritonclient.utils.triton_to_np_dtype(meta_in.datatype)
        print(f'{model}: input {meta_in.name} {list(meta_in.shape)} '
              f'{meta_in.datatype} -> recording {len(entry["requests"])}')

        index = {'input_name': meta_in.name, 'shape': shape,
                 'datatype': meta_in.datatype, 'requests': []}

        for request in entry['requests']:
            array = make_input(shape, dtype, request)

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
            with open(os.path.join(out_dir, f'response_{name}.pb'), 'wb') as f:
                f.write(result.get_response().SerializeToString())

            index['requests'].append({
                'name': name,
                'fill': request.get('fill'),
                'image': request.get('image'),
                'class_count': request['class_count'],
            })
            print(f'  recorded {name} (class_count={request["class_count"]})')

        with open(os.path.join(out_dir, 'index.json'), 'w') as f:
            json.dump(index, f, indent=2, sort_keys=True)

    print('fixtures written to', FIXTURES)


if __name__ == '__main__':
    main()
