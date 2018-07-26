# import package
import tensorflow as tf 
import cifar10_train_test
import cifar10_input_pipeline
import os
import argparse
import tfRecord_input_pipeline

# args parse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/mao/Desktop/example/jupyter_notebook/TFRecord/tfrecord/eval/', help='eval dataset dir')
parser.add_argument('--batch_size', type=int, default=32, help='eval data batch size')
parser.add_argument('--mode', type=str, default='eval', help='train or eval')
parser.add_argument('--checkpoint_dir', type=str, default='./model_1532068180', help='checkpoint dir')
parser.add_argument('--eval_steps', type=int, default=10001, help='eval num step')
FLAGS = parser.parse_args()

# return accuracy
def accuracy(logits, labels):
	# calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
                                          tf.float32),
                               name='accuracy')
    # summary accuracy
    tf.summary.scalar('Accuracy', accuracy)

    return accuracy


def main():
	# get filename
	#filenames = [os.path.join(FLAGS.data_dir, 'test_batch.bin')]
	filenames = [os.path.join(FLAGS.data_dir, tfrecord) for tfrecord in os.listdir(FLAGS.data_dir)]
	# put data pipeline in cpu, for speed up
	with tf.device('/cpu:0'):
		#example_batch = cifar10_input_pipeline.input_pipeline(filenames, batch_size=FLAGS.batch_size, num_epochs=None, mode=FLAGS.mode)
		images, labels = tfRecord_input_pipeline.input_pipeline(filenames, batch_size=FLAGS.batch_size, num_epochs=None, mode=FLAGS.mode)
	# get images and labels batch
	#images = example_batch[0]
	#labels = example_batch[1]
	# get logits
	logits = cifar10_train_test.inference(images, mode='eval')
	# restore trainable variables
	variable_averages = tf.train.ExponentialMovingAverage(0.9)
	variables_to_restore = variable_averages.variables_to_restore()
	saver = tf.train.Saver(variables_to_restore)
	# create session
	sess = tf.Session()
	# get checkpoint dir
	checkpoint_dir = FLAGS.checkpoint_dir
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	saver.restore(sess, ckpt.model_checkpoint_path)
	# create FileWriter writer
	tf.summary.FileWriterCache.clear()
	writer = tf.summary.FileWriter('./accuracy_flowers', sess.graph)
	# get accuracy
	_accuracy = accuracy(logits, labels)
	merged = tf.summary.merge_all()
	# create Coordinator
	coord = tf.train.Coordinator()
	# starts all queue runners collected in grpah
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	try:
		for i in range(FLAGS.eval_steps):
			if not coord.should_stop():
				# sess run merged summary to get bytes of string tensor
				merge_sum = sess.run(merged)
				if i % 100 == 0:
					print('step: %d' % i)
					# write summary to event file
					writer.add_summary(merge_sum, i)
	except tf.errors.OutOfRangeError:
		print('catch OutOfRangeError')
	finally:
		# request that the threads stop, after this is called, calls to should_stop() will return True
		coord.request_stop()
	# wait for threads to terminate
	coord.join(threads)
	# manually flush summary to events in disk
	writer.flush()
	# close writer
	writer.close()
	# close session
	sess.close()
	pass

if __name__ == '__main__':
	main()