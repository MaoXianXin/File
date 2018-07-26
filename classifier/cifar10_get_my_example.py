# import package
import tensorflow as tf 
import os
import argparse
import cv2

# args parse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/mao/Notebooks/cifar10/cifar-10-batches-bin/', help='input dataset dir')
parser.add_argument('--image_height', type=int, default=32, help='image height')
parser.add_argument('--image_width', type=int, default=32, help='image width')
parser.add_argument('--image_depth', type=int, default=3, help='image channels')
parser.add_argument('--label_bytes', type=int, default=1, help='label bytes')
FLAGS = parser.parse_args()

# get single example from filename_path dequeue from filename_queue
def get_my_example(filename_queue):
	# store example
	class CIFAR10Record(object):
		pass
	# initial a store object
	result = CIFAR10Record()
	# info about imgage and label
	label_bytes = FLAGS.label_bytes
	result.height = FLAGS.image_height
	result.width = FLAGS.image_width
	result.depth = FLAGS.image_depth
	# info about image bytes and single example bytes
	image_bytes = result.height * result.width * result.depth
	record_bytes = label_bytes + image_bytes
	# create a FixedLengthRecordReader
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	# returns the next record (key, value) pair produced by a reader
	result.key, value = reader.read(filename_queue)
	# reinterpret the bytes of a string as a vector of numbers
	image_tensor = tf.decode_raw(value, tf.uint8)
	# get label from vector of numbers
	result.label = tf.reshape(
			tf.cast(
				tf.strided_slice(image_tensor, [0], [label_bytes]), tf.int32), shape=[])
	# get image from vector of numbers
	depth_major = tf.reshape(
			tf.strided_slice(image_tensor, [label_bytes],
				[label_bytes + image_bytes]), [result.depth, result.height, result.width])
	# transform CHW to HWC
	result.uint8image = tf.transpose(depth_major, [1,2,0])

	return result

# using opencv view image
def opencv_view(img):
	cv2.imshow('img', img[:,:,::-1])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	# get filename paths
	filenames = [os.path.join(FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
	# get filename queue
	filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
	# get single example from filename_path dequeue from filename_queue
	example = get_my_example(filename_queue)
	# create op to initialize variables
	init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
	# create a session
	sess = tf.Session()
	# run op to initialize variables
	sess.run(init_op)
	# create a new Coordinator
	coord = tf.train.Coordinator()
	# starts all queue runners collected in the graph
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	try:
		print(example.key, example.uint8image, example.label)
		key, img, label = sess.run([example.key, example.uint8image, example.label])
		print(key, img, label)
	except tf.errors.OutOfRangeError:
		print('catch OutOfRangeError')
	finally:
		# request that the threads stop, after this is called, calls to should_stop() will return True
		coord.request_stop()
		print('finish read')
	# using opencv to show the image
	opencv_view(img)
	# wait for threads to terminate
	coord.join(threads)
	# close the session
	sess.close()


if __name__ == '__main__':
	main()