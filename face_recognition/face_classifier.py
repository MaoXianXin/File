# import packages
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import time

path = '/home/mao/Desktop/face_recognition/dataset/'
modelPath = '/home/mao/Desktop/face_recognition/model/'
# read data from cvs file
face_dataFrame = pd.read_csv(path + 'face.csv', header=None)
# transform DataFrame to numpy array
face_ndarray = np.array(face_dataFrame)
# face_encodings
face_encodes = face_ndarray[:, 0:128].astype('float32')
# face_names
face_names = face_ndarray[:, -1]
# map name to id
def names_map_ids(csv_path):
    data = pd.read_csv(csv_path, header=None)
    data = np.array(data)
    
    ids = data[:, 0]
    names = data[:, -1]
    
    names_to_id = {}
    num_class = len(ids)
    for i in range(len(ids)):
        names_to_id[names[i]] = i
    return names_to_id, num_class
name_map_id, num_class = names_map_ids(path + 'names_ids.csv')
# function transform names to ids
def name_to_id(y):
	ids = []
	for i in range(len(y)):
		ids.append(name_map_id[y[i]])
	return ids
# invoking name_to_id function
ids = name_to_id(face_names)
# transform ids to numpy array
ids = np.array(ids, dtype=int)
# make data batch
def train_input_fn(features, labels, batch_size):
	# make data batch from tensor-like array
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	# generate data batch
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	return dataset
# dataset batch object
dataset = train_input_fn(face_encodes, ids, 32)
# iterator object
iterator = dataset.make_initializable_iterator()
# get next batch
get_next = iterator.get_next()
# x is input features, y is input labels
x = tf.placeholder(shape=[None, 128], dtype=tf.float32, name='features')
y = tf.placeholder(shape=[None], dtype=tf.int64, name='labels')
# model structure
dense1 = tf.layers.dense(x, 100, tf.nn.relu, name='fc1')
dense2 = tf.layers.dense(dense1, 50, tf.nn.relu, name='fc2')
dense3 = tf.layers.dense(dense2, 10, tf.nn.relu, name='fc3')
logits = tf.layers.dense(dense3, num_class, name='logits')
# loss function
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
mean_loss = tf.reduce_mean(loss)
# optimizer loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# training operation
train_op = optimizer.minimize(mean_loss)
# create session instance
sess = tf.Session()
# make some initialize
sess.run(iterator.initializer)
sess.run(tf.global_variables_initializer())
# saver object, used to save model
saver = tf.train.Saver()
# start time
start = time.time()
# start training, iteration is 50000
for i in range(10000):
	(features, labels) = sess.run(get_next)
	loss_value, _ = sess.run([mean_loss, train_op], feed_dict={x:features, y:labels})
	print(loss_value)
# end time
end = time.time()
# print consume time
print(end - start)
# save the model, and return model path
save_path = saver.save(sess, modelPath + 'model.ckpt')

# evaluation model
(features, labels) = sess.run(get_next)
# make prediction
prediction = sess.run(logits, feed_dict={x:features, y:labels})
# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.int32))
# print accuracy
print(sess.run(accuracy, feed_dict={x:features, y:labels}))
