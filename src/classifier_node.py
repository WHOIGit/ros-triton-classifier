#!/usr/bin/env python3
import functools

import numpy as np
import rospy

# cv2 unused, but import required to solve unreported exception
# See https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
import cv2
from cv_bridge import CvBridge
from PIL import Image as PilImage
from sensor_msgs.msg import Image

from triton_api import Model, ImageInput, ClassificationOutput, ScalingMode, initialize_model
from triton_classifier.msg import Classification, ObjectHypothesisWithClassName



def on_image(model, output_name, class_pub, image_msg):
    # Use the cv_bridge to convert to an OpenCV image object
    img = CvBridge().imgmsg_to_cv2(image_msg)
    # Convert the OpenCV image to a PIL image
    pil_image = PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Ask the classifier to infer result
    try:
        result = model.infer(pil_image)
    except Exception as e:
        rospy.logerr('Error getting classification from Triton: %s', e)
        raise e

    # Null-check response
    classifications = getattr(result, output_name)
    if len(classifications) < 1:
       raise ValueError('Unexpected result from classifier', repr(result))

    # Format message and publish
    classification = Classification()
    classification.header = image_msg.header
    classification.results = []

    for r in classifications:
        h = ObjectHypothesisWithClassName()
        h.class_name = r.class_name
        h.score = r.score
        classification.results.append(h)

    class_pub.publish(classification)


def main():
    rospy.init_node('classifier', anonymous=True)

    model = initialize_model(rospy.get_param('~triton_server_url'), rospy.get_param('~classifier_model'))

    # Bind by the model's own tensor names; they are not always input/output.
    # Assigning an attribute that is not a real tensor name would silently skip
    # bind(), leaving the default TensorInput in place.
    input_name = model.metadata.inputs[0].name
    output_name = model.metadata.outputs[0].name
    setattr(model, input_name, ImageInput(scaling=ScalingMode.INCEPTION))
    setattr(model, output_name, ClassificationOutput(classes=3))

    image_topic = rospy.get_param('~image_topic')

    # Advertise that we will publish a "/class" subtopic of the image topic
    class_pub = rospy.Publisher(
        image_topic + '/class',
        Classification,
        queue_size=1
    )

    # Subscribe to the raw image data. image_transport publishes the raw
    # sensor_msgs/Image on the base topic itself -- only the other transports
    # get "<base topic>/<transport name>" subtopics -- so do not append /raw.
    rospy.Subscriber(
        image_topic,
        Image,
        functools.partial(on_image, model, output_name, class_pub)
    )

    rospy.spin()


if __name__ == '__main__':
    main()
