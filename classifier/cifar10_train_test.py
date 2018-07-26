# import package
import tensorflow as tf
import os
import cifar10_input_pipeline
import time
import argparse
import tfRecord_input_pipeline

# args parse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/mao/Desktop/example/jupyter_notebook/TFRecord/tfrecord/train/', help='input dataset dir')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of input data')
parser.add_argument('--mode', type=str, default='train', help='train or eval')
parser.add_argument('--initial_learning_rate', type=float, default=0.02, help='initial learning rate')
parser.add_argument('--decay_steps', type=int, default=10000, help='decay learning rate every steps')
parser.add_argument('--decay_rate', type=float, default=0.3, help='learning rate decay rate')
parser.add_argument('--train_step', type=int, default=100001, help='training step')
parser.add_argument('--save_every_step', type=int, default=10000, help='save checkpoint every num step')
parser.add_argument('--NUM_CLASSES', type=int, default=5, help='number of classes')
FLAGS = parser.parse_args()

# initialize variables, such as kernel, weights, bias
def variable_initializer(name, shape, _type, initializer, wd):
	# create variable in cpu
	with tf.device('/cpu:0'):
		var = tf.get_variable(name,
							shape,
							_type,
							initializer)
	# judge whether there is wd
	if wd is not None:
		# calculate weight decay
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		# add weight decay to losses collection
		tf.add_to_collection('losses', weight_decay)

	return var

# conv layer, using relu activation
def conv_relu(inputs, kernel_shape, bias_shape, activation):
	# initialize kernel
	kernel = variable_initializer('weights',
								kernel_shape,
								tf.float32,
								initializer=tf.random_normal_initializer(stddev=0.05),
								wd=None)
	# initialize bias
	bias = variable_initializer('bias',
								bias_shape,
								tf.float32,
								initializer=tf.zeros_initializer(),
								wd=None)
	# conv the input
	conv = tf.nn.conv2d(inputs, kernel, [1,1,1,1], padding='SAME', name='conv')
	# using bn
	conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, scope='bn')
	# using activation
	conv = activation(tf.nn.bias_add(conv, bias), name='relu')

	return conv

# fully-connected layer, using relu activation
def fc_relu(inputs, weights_shape, bias_shape, activation, wd):
	# initialize weight
	weights = variable_initializer('weights',
									weights_shape,
									tf.float32,
									initializer=tf.random_normal_initializer(stddev=0.05),
									wd=wd)
	# initialize bias
	bias = variable_initializer('bias',
								bias_shape,
								tf.float32,
								initializer=tf.zeros_initializer(),
								wd=None)
	# weight sum
	fc = tf.matmul(inputs, weights, name='fc')
	# using bn
	fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True, scope='bn')
	# judge whether it's the final layer
	if activation is not None:
		fc = activation(tf.nn.bias_add(fc, bias), name='relu')
	else:
		fc = tf.nn.bias_add(fc, bias, name='logits')

	return fc

# images shape can be [batch_size, height, width, 3]
def inference(images, mode='train'):
	# conv1
    with tf.variable_scope('conv1'):
    	conv1 = conv_relu(images, [3,3,3,32], [32], tf.nn.relu)
    # pool1
    pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, name='norm1')
    # conv2
    with tf.variable_scope('conv2'):
    	conv2 = conv_relu(norm1, [3,3,32,64], [64], tf.nn.relu)
    # pool2
    pool2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool2')
    # norm2
    norm2 = tf.nn.lrn(pool2, name='norm2')
    # conv3
    with tf.variable_scope('conv3'):
    	conv3 = conv_relu(norm2, [3,3,64,128], [128], tf.nn.relu)
    # pool3
    pool3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool3')
    # norm3
    norm3 = tf.nn.lrn(pool3, name='norm3')
    # conv4
    with tf.variable_scope('conv4'):
    	conv4 = conv_relu(norm3, [1,1,128,8], [8], tf.nn.relu)
    # pool4
    pool4 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool4')
    # norm4
    norm4 = tf.nn.lrn(pool4, name='norm4')
    # get batch size
    batch_size = images.get_shape()[0].value
    # flatten the norm3
    flatten = tf.reshape(norm4, shape=[batch_size, -1])
    # input layer neurons number
    dim = flatten.get_shape()[1].value
    # fc1
    with tf.variable_scope('fc1'):
    	fc1 = fc_relu(flatten, [dim, 1024], [1024], tf.nn.relu, 0.004)
    # dropout1
    drop1 = tf.layers.dropout(fc1, training=mode == 'train')
    # fc2
    with tf.variable_scope('fc2'):
    	fc2 = fc_relu(drop1, [1024, 384], [384], tf.nn.relu, 0.004)
    # dropout2
    drop2 = tf.layers.dropout(fc2, training=mode == 'train')
    # logits layer, output layer without acitvation
    with tf.variable_scope('softmax_linear'):
    	logits = fc_relu(drop2, [384, FLAGS.NUM_CLASSES], [FLAGS.NUM_CLASSES], None, None)

    print('inference sucess')

    return logits

# caculate total_loss
def loss(logits, labels):
	# calculate cross_entropy, logits=[batch_size, classes], labels=[batch_size]
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                  name='cross_entropy')
    # calculate loss
    loss = tf.reduce_mean(cross_entropy, name='loss')
    # add loss to losses collection
    tf.add_to_collection('losses', loss)

    print('loss sucess')
    # calculate total loss
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss

# add moving average
def moving_average(total_loss):
	# maintains moving averages of variables by employing an exponential decay
	loss_average = tf.train.ExponentialMovingAverage(0.9, name='moving_average')
	# get all loss tensor in list
	losses = tf.get_collection('losses')
	# create loss replicate
	loss_average_op = loss_average.apply(losses + [total_loss])
	# summary all loss, and replicate loss
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name + '(raw)', l)
		tf.summary.scalar(l.op.name, loss_average.average(l))

	return loss_average_op

# return training op
def train(total_loss, global_step):
	# using learning rate exponential decay when training
	lr = tf.train.exponential_decay(
		learning_rate=FLAGS.initial_learning_rate,
		global_step=global_step,
		decay_steps=FLAGS.decay_steps,
		decay_rate=FLAGS.decay_rate,
		staircase=True,
		name='learning_rate_decay')
	# add moving average to loss
	loss_average_op = moving_average(total_loss)
	# return train_op, after loss_average_op is run
	with tf.control_dependencies([loss_average_op]):
		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
		train_op = opt.minimize(total_loss, global_step)
	# moving average all trainable variables
	variable_average = tf.train.ExponentialMovingAverage(0.9, global_step)
	with tf.control_dependencies([train_op]):
		variable_average_op = variable_average.apply(tf.trainable_variables())
	print('train sucess')

	return variable_average_op

# calculate accuracy, use train data batch
def accuracy(logits, labels):
	# calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
                                          tf.float32),
                               name='accuracy')
    # summary accuracy
    tf.summary.scalar('Accuracy', accuracy)

    return accuracy

def main():
	# get filenames
	#filenames = [os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
	filenames = [os.path.join(FLAGS.data_dir, tfrecord) for tfrecord in os.listdir(FLAGS.data_dir)]
	# get timestamp, using for checkpoint and events
	timestamp = time.time()
	timestamp = str(int(timestamp))
	# checkpoint save dir
	model_path = './model_' + timestamp + '/model.ckpt'
	# clear summary cache
	tf.summary.FileWriterCache.clear()
	# put data batch read in cpu, for speed up
	with tf.device('/cpu:0'):
		#example_batch = cifar10_input_pipeline.input_pipeline(filenames, batch_size=FLAGS.batch_size, num_epochs=None, mode=FLAGS.mode)
		images, labels = tfRecord_input_pipeline.input_pipeline(filenames, batch_size=FLAGS.batch_size, num_epochs=None, mode=FLAGS.mode)
	# get images and labels
	#images = example_batch[0]
	#labels = example_batch[1]
	# calculate logits
	logits = inference(images, mode='train')
	# calculate total loss
	total_loss = loss(logits, labels)
	# get global step
	global_step = tf.train.get_or_create_global_step()
	# get train_op
	train_op = train(total_loss, global_step)
	# get accuracy
	_accuracy = accuracy(logits, labels)
	# create session
	sess = tf.Session()
	# create checkpoint saver
	saver = tf.train.Saver()
	# create summary FileWriter writer
	writer = tf.summary.FileWriter('./event_' + timestamp, sess.graph)
	# create init op
	init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
	# run op to initialize all variables
	sess.run(init_op)
	# merge all summary, in order to write to event file
	merged = tf.summary.merge_all()
	# create Coordinator
	coord = tf.train.Coordinator()
	# starts all queue runners collected in the graph
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	try:
	    for i in range(FLAGS.train_step):
	        if not coord.should_stop():
	        	# run the merged summary, to get bytes of string tensor
	            _, merged_sum, _totoal_loss = sess.run([train_op, merged, total_loss])
	            print(_totoal_loss)
	            if i % 100 == 0:
	            	# writer summary to event file
	            	writer.add_summary(merged_sum, i)
	            	print('step: %d' % i)
	            if i % FLAGS.save_every_step == 0:
	            	# save the checkpoint file
	            	saver.save(sess, model_path, i)
	except tf.errors.OutOfRangeError:
	    print('catch OutOfRangeError')
	finally:
		# request that the threads stop, after this is called, calls to should_stop() will return True
	    coord.request_stop()

	# wait for threads to terminate
	coord.join(threads)
	# manually flush summary to event file
	writer.flush()
	# close writer
	writer.close()
	# close session
	sess.close()

if __name__ == '__main__':
	main()