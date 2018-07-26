#############
# Tasks     #
#############

get_an_example_binary:
		python3 cifar10_get_my_example.py

get_an_example_batch_binary:
		python3 cifar10_input_pipeline.py

train:
		python3 cifar10_train_test.py

eval:
		python3 cifar10_eval_test.py

tfrecord:
		python3 /home/mao/Documents/Github/models/research/slim/datasets/download_and_convert_flowers.py \
				--dataset_dir /home/mao/Documents/datasets/ \
				--dataset_folder_name flower_photos \
				--output_dataset_dir /home/mao/Desktop/example/jupyter_notebook/TFRecord/
