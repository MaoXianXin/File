# import package
import tensorflow as tf 
import os
import cifar10_get_my_example
import argparse
import cv2

# args parse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/mao/Desktop/example/jupyter_notebook/TFRecord/tfrecord/eval/', help='input dataset dir')
parser.add_argument('--image_final_height', type=int, default=250, help='final image height')
parser.add_argument('--image_final_width', type=int, default=250, help='final image height')
parser.add_argument('--image_depth', type=int, default=3, help='image channels')
parser.add_argument('--min_after_dequeue', type=int, default=3000, help='minimum number elements in the queue after a dequeue')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of example batch')
parser.add_argument('--image_resize_height', type=int, default=300, help='resize image height to 300')
parser.add_argument('--image_resize_width', type=int, default=300, help='resize image width to 300')
FLAGS = parser.parse_args()

# using opencv view image
def opencv_view(img):
	cv2.imshow('img', img[:,:,::-1])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# preprocess train image
def preprocess_train_image(image):
	# cast image type to uint8
    float_image = image
    # final image height and width
    height = FLAGS.image_final_height
    width = FLAGS.image_final_width
    float_image = tf.image.resize_images(float_image, [FLAGS.image_resize_height,FLAGS.image_resize_width])
    # random crop image from 32*32*3 to 24*24*3
    distorted_image = tf.random_crop(float_image, [height, width, FLAGS.image_depth])
    # random flip
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # random brightness adjust x - 0.2 <= y <= x + 0.2
    distorted_image = tf.image.random_brightness(distorted_image,
                                                    max_delta=0.2)
    # random contrast
    distorted_image = tf.image.random_contrast(distorted_image,
                                                  lower=0.2, upper=1.8)
    # linearly scale image to have zero mean and unit norm
    float_image = tf.image.per_image_standardization(distorted_image)
    
    return float_image

# preprocess eval image
def preprocess_eval_image(image):
	# cast image type to uint8
    float_image = image
    # final image height and width
    height = FLAGS.image_final_height
    width = FLAGS.image_final_width
    float_image = tf.image.resize_images(float_image, [FLAGS.image_resize_height,FLAGS.image_resize_width])
    # resize image and central crop
    distorted_image = tf.image.resize_image_with_crop_or_pad(float_image, height, width)
    # linearly scale image to have zero mean and unit norm
    float_image = tf.image.per_image_standardization(distorted_image)
    
    return float_image

def get_my_example(filename_queue):
    feature = {
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64)
    }
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    features = tf.parse_single_example(serialized_example, features=feature)
    
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    label = tf.cast(features['image/class/label'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    
    image = tf.reshape(image, [height, width, 3])
    
    return image, label

# get example batch
def input_pipeline(filenames, batch_size, num_epochs=None, mode='train'):
	# get filename_queue
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
	# get single example
	image, label = get_my_example(filename_queue)
	# judge training mode or eval mode
	if mode == 'train':
		float_image = preprocess_train_image(image)
	else:
		float_image = preprocess_eval_image(image)
	# minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements
	min_after_dequeue = FLAGS.min_after_dequeue
	# controls the how long the prefetching is allowed to grow the queues, maximum number of elements in the queue
	capacity = min_after_dequeue + 3 * batch_size
	# get example batch
	example_batch = tf.train.shuffle_batch(
		[float_image, label], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue, num_threads=4)

	return example_batch

def main():
	# get filename_paths
	filenames = [os.path.join(FLAGS.data_dir, tfrecord) for tfrecord in os.listdir(FLAGS.data_dir)]
	# get example batch
	images, labels = input_pipeline(filenames, batch_size=FLAGS.batch_size, num_epochs=None)
	# create initial op
	init_op = tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer())
	# create session
	sess = tf.Session()
	# run op to initial variables
	sess.run(init_op)
	# create Coordinator
	coord = tf.train.Coordinator()
	# starts all queue runners collected in the graph
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	try:
		print(images, labels)
		images, labels = sess.run([images, labels])
		print(images, labels)
	except tf.errors.OutOfRangeError:
		print('catch OutOfRangeError')
	finally:
		# request that the threads stop, after this is called, calls to should_stop() will return True
		coord.request_stop()
		print('finish read')
	# using opencv to show the image
	opencv_view(images[0])
	# wait for threads to terminate
	coord.join(threads)
	# close the session
	sess.close()

if __name__ == '__main__':
	main()