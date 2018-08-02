# import packages
import numpy as np
import pandas as pd
import face_recognition
import os

path = '/home/mao/Desktop/face_recognition/dataset/'
try:
    os.remove(path + 'face.csv')
    os.remove(path + 'names_ids.csv')
except NotADirectoryError:
    pass
# get face_names
face_names = os.listdir(path)
# store face_encodes to x, face_names to y
x = []
y = []
# store face_name and face_id
names_and_ids = []

for i in range(len(face_names)):
	# get face_name, folder name is face_name
    face_name = face_names[i]
    names_and_ids.append([i, face_name])
    # get image files under folder face_name
    img_file = os.listdir(path + face_name)
    for j in range(len(img_file)):
    	# face image path
        file_path = path + face_name + '/' + img_file[j]
        print(j, file_path)
        # load face
        face_load = face_recognition.load_image_file(file_path)
        try:
        	# encoding face
            face_encode = face_recognition.face_encodings(face_load)[0]
            x.append(face_encode)
            y.append(face_name)
        except IndexError:
        	# if encode fail, print this
            print('ignore')
# transform list x and y to numpy array
x = np.asarray(x)
y = np.asarray(y)
# expend y to 2-D array
y_expend = np.expand_dims(y, axis=0)
# concatenate x and y
data = np.concatenate((x, y_expend.T), axis=1)
# create DataFrame, in order to write to csv file
pd_dataFrame = pd.DataFrame(data=data)
pd_dataFrame1 = pd.DataFrame(data=names_and_ids)
# write DataFrame to csv file

pd_dataFrame.to_csv(path + '/face.csv', header=False, index=False)
names_and_ids = np.asarray(names_and_ids)
pd_dataFrame1.to_csv(path + 'names_ids.csv', header=False, index=False)