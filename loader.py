import numpy as np
import os
import random
from PIL import Image

def vectorized_result(y):
	f_array = np.zeros((2, 1))
	f_array[y] = 1
	return f_array

def load_images(img_type):
	all_train_images = os.listdir('/home/tushar/Downloads/CarData/{}/'.format(img_type))
	random.shuffle(all_train_images)
	image_features, image_labels = [], []

	for im in all_train_images:
		if 'pos' in im:
			ans = 1
		elif 'neg' in im:
			ans = 0
		img = Image.open('/home/tushar/Downloads/CarData/{}/{}'.format(img_type, im))
		img_array = np.asarray(img)
		img_array = img_array.reshape(4000,1)
		img_label = vectorized_result(ans)
		image_features.append(img_array)
		image_labels.append(img_label)	

	train_features, train_labels = image_features[:-550], image_labels[:-550]
	test_features, test_labels = image_features[-550:], image_labels[-550:]

	return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels) 

