#!/usr/bin/env python3
'''
Publishes Triton inference results as standard vision_msgs.

Two tasks are supported, selected with the ~task parameter:

  classification -> vision_msgs/Classification2D on <image_topic>/classification
  detection      -> vision_msgs/Detection2DArray on <image_topic>/detections

vision_msgs identifies classes by a numeric id only, so the id -> name mapping
travels out of band: this node loads the class names, puts them on the
parameter server, and advertises where it put them with a latched
vision_msgs/VisionInfo on <image_topic>/vision_info. That is what VisionInfo is
for -- its own definition recommends "an XML string on the ROS parameter
server" -- and it matches the convention used by ros_deep_learning, which
publishes class_labels_<hash> alongside a VisionInfo pointing at it.
'''

import functools
import hashlib

import numpy as np
import rospy

# cv2 unused, but import required to solve unreported exception
# See https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
import cv2
from cv_bridge import CvBridge
from PIL import Image as PilImage
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Classification2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    VisionInfo,
)

from triton_api import (
    ClassificationOutput,
    DetectionOutput,
    ImageInput,
    ScalingMode,
    initialize_model,
)


def load_class_labels():
    '''
    Read the class names, as either a list of names directly on the parameter
    server or a path to a file with one name per line. Returns [] if neither is
    configured, which is legal -- consumers then only get numeric ids.
    '''
    labels = rospy.get_param('~class_labels', None)
    if isinstance(labels, list):
        return [str(label) for label in labels]

    path = rospy.get_param('~class_labels_file', None)
    if path:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    return []


def publish_class_labels(model_name, labels):
    '''
    Put the id -> name map on the parameter server and return the fully
    resolved parameter name, for VisionInfo.database_location.

    The name is keyed by a hash of the model and labels so that two nodes
    serving different models cannot quietly overwrite each other's map.
    '''
    digest = hashlib.sha1(
        ('\n'.join([model_name] + labels)).encode('utf-8')).hexdigest()[:8]
    key = rospy.resolve_name('~class_labels_%s' % digest)

    # A dict keyed by str(id): ROS parameters are XML-RPC structs, so the keys
    # have to be strings even though the ids are numeric.
    rospy.set_param(key, {str(i): name for i, name in enumerate(labels)})
    return key


def on_image(model, image_input, output_name, task, publisher, image_msg):
    # Use the cv_bridge to convert to an OpenCV image object
    img = CvBridge().imgmsg_to_cv2(image_msg)
    # Convert the OpenCV image to a PIL image
    pil_image = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    try:
        result = model.infer(pil_image)
    except Exception as e:
        rospy.logerr('Error getting inference from Triton: %s', e)
        raise e

    results = getattr(result, output_name)

    if task == 'detection':
        publisher.publish(
            build_detections(results, image_input, pil_image.size, image_msg))
    else:
        publisher.publish(build_classification(results, image_msg))


def build_detections(detections, image_input, source_size, image_msg):
    '''Convert triton_api Detections into a vision_msgs/Detection2DArray.'''
    array = Detection2DArray()
    array.header = image_msg.header
    array.detections = []

    for d in detections:
        # Detections come back in the model's input frame; put them back on the
        # source image so the boxes mean something to a consumer.
        x1, y1, x2, y2 = image_input.to_source_box((d.x1, d.y1, d.x2, d.y2),
                                                   source_size)

        detection = Detection2D()
        detection.header = image_msg.header
        detection.bbox.center.x = (x1 + x2) / 2.0
        detection.bbox.center.y = (y1 + y2) / 2.0
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = d.class_id
        hypothesis.score = d.score
        detection.results = [hypothesis]

        array.detections.append(detection)

    return array


def build_classification(classifications, image_msg):
    '''
    Convert triton_api Classifications into a vision_msgs/Classification2D.

    vision_msgs says score "should lie in the range [0-1]". The node asks
    ClassificationOutput to apply a softmax over the model's complete logit
    vector by default, so that holds; with ~activation='' the scores are the
    server's raw logits instead and it does not.
    '''
    message = Classification2D()
    message.header = image_msg.header
    message.results = []

    for c in classifications:
        hypothesis = ObjectHypothesis()
        hypothesis.id = c.class_id
        hypothesis.score = c.score
        message.results.append(hypothesis)

    return message


def main():
    rospy.init_node('classifier', anonymous=True)

    task = rospy.get_param('~task', 'classification')
    if task not in ('classification', 'detection'):
        raise ValueError("~task must be 'classification' or 'detection', "
                         'got %r' % task)

    model_name = rospy.get_param('~classifier_model')
    model = initialize_model(rospy.get_param('~triton_server_url'), model_name)

    # Bind by the model's own tensor names; they are not always input/output.
    input_name = model.metadata.inputs[0].name
    output_name = model.metadata.outputs[0].name

    labels = load_class_labels()

    if task == 'detection':
        image_input = ImageInput(scaling=ScalingMode.NORM, letterbox=True,
                                 layout=rospy.get_param('~layout', None))
        output = DetectionOutput(
            confidence=rospy.get_param('~confidence_threshold', 0.25),
            iou=rospy.get_param('~iou_threshold', 0.45),
            labels=labels or None)
    else:
        image_input = ImageInput(scaling=ScalingMode.INCEPTION,
                                 layout=rospy.get_param('~layout', None))
        # Normalize locally by default so that `score` is a probability, as
        # vision_msgs expects. Set ~activation to '' for the server's raw
        # logits, or to 'sigmoid' for a multi-label model.
        activation = rospy.get_param('~activation', 'softmax') or None
        output = ClassificationOutput(
            classes=rospy.get_param('~classes', 3),
            activation=activation,
            labels=labels or None)

    setattr(model, input_name, image_input)
    setattr(model, output_name, output)

    image_topic = rospy.get_param('~image_topic')

    if task == 'detection':
        publisher = rospy.Publisher(image_topic + '/detections',
                                    Detection2DArray, queue_size=1)
    else:
        publisher = rospy.Publisher(image_topic + '/classification',
                                    Classification2D, queue_size=1)

    # Advertise where the id -> name map lives. Latched, so that subscribers
    # which come up after us still receive it.
    info = VisionInfo()
    info.header.stamp = rospy.Time.now()
    info.method = model_name
    info.database_location = publish_class_labels(model_name, labels) \
        if labels else ''
    info.database_version = 0

    info_publisher = rospy.Publisher(image_topic + '/vision_info', VisionInfo,
                                     queue_size=1, latch=True)
    info_publisher.publish(info)

    if not labels:
        rospy.logwarn('No ~class_labels or ~class_labels_file configured; '
                      'publishing numeric class ids only')

    # Subscribe to the raw image data. image_transport publishes the raw
    # sensor_msgs/Image on the base topic itself -- only the other transports
    # get "<base topic>/<transport name>" subtopics -- so do not append /raw.
    rospy.Subscriber(
        image_topic,
        Image,
        functools.partial(on_image, model, image_input, output_name, task,
                          publisher)
    )

    rospy.spin()


if __name__ == '__main__':
    main()
