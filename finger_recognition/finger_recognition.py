# import packages
import tensorflow as tf
import cv2
import numpy as np 
import time
import argparse
import imutils
import dlib
from imutils.video import VideoStream
import pandas as pd

class FingerClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = './export_pb_graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
    
    
    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num

class_id_to_name = {'1': 'ok'}
person = FingerClassifier()

vs = VideoStream(src=0).start()
time.sleep(3.0)
flag = 1

while True:
    if flag == -1:
        img = vs.read()
        #img = cv2.resize(img, (700, 700))
        rboxes, rscores, rclasses, rnum = person.get_classification(img[:,:,::-1])
        h = img.shape[0]
        w = img.shape[1]
        for i in range(rboxes.shape[1]):
            if rscores[0][i] > 0.5:
                sx = int(rboxes[0][i][1] * w)
                sy = int(rboxes[0][i][0] * h)
                ex = int(rboxes[0][i][3] * w)
                ey = int(rboxes[0][i][2] * h)
                key = str(int(rclasses[0][i]))
                name = class_id_to_name[key]
                cv2.rectangle(img, (sx, sy), (ex, ey), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (sx-5, ey + 26), font, 0.8, (0, 0, 255), 1)
                
                timestamp = time.time()
        
        cv2.imshow('finger', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
    
    flag = flag * -1

cv2.destroyAllWindows()
vs.stop()