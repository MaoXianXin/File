# import packages
import tensorflow as tf 
import cv2
import numpy as np 
import time
import face_recognition
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
from imutils.video import VideoStream
import pandas as pd

prototxt_path = "/home/mao/Desktop/face_recognition/preModel/deploy.prototxt.txt"
model_path = "/home/mao/Desktop/face_recognition/preModel/vgg16_300x300_ssd_iter_140000.caffemodel"
shape_predictor_path = "/home/mao/Desktop/face_recognition/preModel/shape_predictor_68_face_landmarks.dat"
confidence = 0.5

path = '/home/mao/Desktop/face_recognition/dataset/'
modelPath = '/home/mao/Desktop/face_recognition/model/'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
predictor = dlib.shape_predictor(shape_predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=200)

cls_graph = tf.get_default_graph()
with cls_graph.as_default():
    graph_def = tf.GraphDef()
    model_path = modelPath + 'frozen-graph.pb'
    with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=cls_graph, 
                     config=config)
    logits = cls_graph.get_tensor_by_name('logits/BiasAdd:0')
    inputs = cls_graph.get_tensor_by_name('features:0')


vs = VideoStream(src=0).start()
time.sleep(2.0)
flag = 1

def ids_map_names(csv_path):
    data = pd.read_csv(csv_path, header=None)
    data = np.array(data)
    
    ids = data[:, 0]
    names = data[:, -1]
    
    id_to_name = {}
    for i in range(len(ids)):
        id_to_name[i] = names[i]
    return id_to_name

id_to_name = ids_map_names(path + 'names_ids.csv')

while True:
    if flag == -1:

        img = vs.read()
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        face_locations = []

        for i in range(0, detections.shape[2]):
            confidence_pred = detections[0,0,i,2]

            if confidence_pred > confidence:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype('int')
                face_locations.append((startX, startY, endX, endY))

        #print(len(face_locations))
        if len(face_locations) == 0:
            cv2.imshow('face', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            continue
            #cv2.destroyAllWindows()

        for i in range(len(face_locations)):
            (startX, startY, endX, endY) = face_locations[i]
            faceAligned = fa.align(img, img, dlib.rectangle(startX, startY, endX, endY))
            face_array = faceAligned[:,:,::-1]
            #face_array = face_array[startY:endY, startX:endX]

            face_encode = face_recognition.face_encodings(face_array)
            face_encode = np.array(face_encode)
            #print(face_encode.shape)

            if face_encode.shape[0] == 0:
                cv2.rectangle(img, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                #cv2.imshow('face', img)
                #key = cv2.waitKey(1) & 0xFF

                #if key == ord("q"):
                    #break
                continue
                #cv2.destroyAllWindows()

            x = tf.placeholder(shape=[None, 128], dtype=tf.float32, name='features')
            y = tf.placeholder(shape=[None], dtype=tf.int64, name='labels')
            logits_value = sess.run(logits, feed_dict={inputs:face_encode})
            probability = sess.run(tf.nn.softmax(logits_value))

            id_num = list(np.argmax(logits_value, axis=1))
            if probability[0][id_num[0]] < 0.5:
                name = "unknown"
            else:
                name = id_to_name[id_num[0]]

            cv2.rectangle(img, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (startX-5, endY + 26), font, 0.8, (0, 0, 255), 1)
        
        cv2.imshow('face', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    
    flag = flag * -1

sess.close()
cv2.destroyAllWindows()
vs.stop()