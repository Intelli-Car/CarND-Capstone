from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2
import time
import os


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frozen_models', 'sim')
# GRAPH_FILE = os.path.join(base_dir, 'ssd_inception_v2_coco_2017_11_17', 'frozen_inference_graph.pb')
# GRAPH_FILE = os.path.join(base_dir, 'ssd_mobilenet_v2_coco_2018_03_29', 'frozen_inference_graph.pb')
GRAPH_FILE = os.path.join(base_dir, 'faster_rcnn_resnet101_coco_2018_01_28', 'frozen_inference_graph.pb')


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        detection_graph = self.load_graph(GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=detection_graph)


    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # Load a sample image.
        image_np = np.expand_dims(np.asarray(image), 0)
        result = TrafficLight.UNKNOWN

        # Actual detection.
        t0 = time.time()
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})
        t1 = time.time()
        time_diff = (t1 - t0) * 1000

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8 #faster rcnn
        # confidence_cutoff = 0.6 #ssd inception
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        i = 0
        result_str = 'unknown'
        score = 0
        if boxes.shape[i] > 0:
            if classes[i] == 1: #'Green':
                result = TrafficLight.GREEN
                result_str = 'Green'
            elif classes[i]  == 2: #'Red':
                result = TrafficLight.RED
                result_str = 'Red'
            elif classes[i]  == 3: #'Yellow':
                result = TrafficLight.YELLOW
                result_str = 'Yellow'

            score = scores[i]
        rospy.logwarn('{}, score {:.4f}, time {:.1f} ms'.format(result_str, score, time_diff))

        return result
