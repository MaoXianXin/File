import tensorflow as tf 
import numpy as np 
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./test.jpg',
	help='image path')
FLAGS = parser.parse_args()

def opencv_view(image):
	"""
		image: is RGB channel
	"""
	cv2.imshow('img', image[:,:,::-1])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

class ImageReader(object):
	"""
		helper class the provides tensorflow image coding utilities
	"""
	def __init__(self):
		# initializes function that decodes RGB jpeg data
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg, 
						 feed_dict = {self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def distort_color(image):
	with tf.device('/cpu:0'):
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image_distorted = tf.image.random_brightness(image, 0.5)
		image_distorted = tf.image.random_contrast(image_distorted, 0.1, 0.6)
		image_distorted = tf.image.random_hue(image_distorted, 0.3)
		image_distorted = tf.image.adjust_gamma(image_distorted, gamma=2, gain=1)
		image_distorted = tf.image.random_saturation(image_distorted, 0.4, 0.9)
	return tf.clip_by_value(image_distorted, 0.0, 1.0)

def main():
	image_reader = ImageReader()

	image_data = tf.gfile.FastGFile(FLAGS.img_path, 'rb').read()

	with tf.Session() as sess:
		with tf.device('/cpu:0'):
			image_array = image_reader.decode_jpeg(sess, image_data)
			image_distorted = sess.run(distort_color(image_array))

	with tf.device('/cpu:0'):
		opencv_view(image_array)
		opencv_view(image_distorted)

	print(image_array.shape)
	print(image_distorted.shape)
	sess.close()

if __name__ == '__main__':
	main()