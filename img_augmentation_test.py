import tensorflow as tf 
import matplotlib.pyplot as plt 

image_path = ['./test.jpg']

filename_queue = tf.train.string_input_producer(image_path)

img_reader = tf.WholeFileReader()

_, image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_jpeg(image_jpg)

image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

image_random_brightness = tf.image.random_brightness(image_decode_jpeg, 0.5)

image_random_contrast = tf.image.random_contrast(image_decode_jpeg, 0.1, 0.6)

image_random_hue = tf.image.random_hue(image_decode_jpeg, 0.3)

image_random_gamma = tf.image.adjust_gamma(image_decode_jpeg, gamma=2, gain=1)

image_random_saturation = tf.image.random_saturation(image_decode_jpeg, 0.4, 0.9)

plt.figure()
plt.subplot(231)
plt.imshow(sess.run(image_random_brightness))
plt.title('random_brightness')

plt.figure()
plt.subplot(232)
plt.imshow(sess.run(image_random_contrast))
plt.title('random_contrast')

plt.figure()
plt.subplot(233)
plt.imshow(sess.run(image_random_hue))
plt.title('random_hue')

plt.figure()
plt.subplot(234)
plt.imshow(sess.run(image_random_gamma))
plt.title('random_gamma')

plt.figure()
plt.subplot(235)
plt.imshow(sess.run(image_random_saturation))
plt.title('random_saturation')

plt.figure()
plt.subplot(236)
plt.imshow(sess.run(image_decode_jpeg))
plt.title('original_image')

plt.show()

coord.request_stop()
coord.join(threads)
sess.close()