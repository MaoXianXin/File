import os

image_folder_dir = "/home/mao/Pictures/test_image1"

image_paths = [os.path.join(image_folder_dir, image_name) for image_name in os.listdir(image_folder_dir)]

count = 0

for image_path in image_paths:
	count += 1
	number_zeros = 6 - len(str(count))
	os.rename(image_path, image_folder_dir + '/' + '0'*number_zeros + str(count) + '.jpg')