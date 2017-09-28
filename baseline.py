import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.datasets import mnist
from generate_datasets import *


if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print('There are %d train and %d test data'%(len(x_train),len(x_test)))
	n_train, n_x, n_y = x_train.shape
	print('Each image has shape %d by %d' %(n_x, n_y))
	#### model params
	f_length = 120
	f_width = 120
	### feature map using encoder
	predictor = Sequential()
	# this is the size of our encoded representations
	encoding_dim_1 = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
	output_dim = 5
	# this is our input placeholder
	#input_img = Input(shape=(f_length*f_width,))
	## hidden layers
	predictor.add(Dense(encoding_dim_1, input_dim=f_length*f_width, activation='sigmoid'))
	predictor.add(Dense(32, input_dim=encoding_dim_1, activation='sigmoid'))
	encoding_dim_2 = 8
	predictor.add(Dense(encoding_dim_2, input_dim=32, activation='selu'))
	#decoded = Dense(f_length*f_width, activation='sigmoid')(encoded)
	## output layers
	predictor.add(Dense(output_dim, input_dim=encoding_dim_2, activation='softmax'))

	###
	predictor.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	### algorithm params
	image_source = x_train[:100]
	max_n_images = 5
	max_overlap_x = 1
	max_overlap_y = 1
	batch_size = 10
	total_n_samples = 1000

	data_gen = mnist_clustering_generator_for_pred(f_length, f_width, 
								   image_source, max_n_images, 
								   max_overlap_x, max_overlap_y, 
								   batch_size, total_n_samples)

	predictor.fit_generator(data_gen,
			steps_per_epoch=total_n_samples/batch_size, epochs=20, verbose=1)
	
	#### create test data
	data_tt, labels_tt = mnist_clustering_dataset_for_pred(f_length, f_width, 
                               image_source, max_n_images, 
                               max_overlap_x, max_overlap_y, 100)
	### get test error
	print(predictor.evaluate(data_tt, labels_tt, verbose=1))
	predictions = predictor.predict_on_batch(data_tt)
	for idx in range(len(labels_tt)):
		print('prediction', np.argmax(predictions[idx]),'label',np.argmax(labels_tt[idx,:]))
	#print(predictor.get_layer(index=1).get_weights())
	# weights1, bias1 = predictor.get_layer(index=1).get_weights()
# 	print('Shape of first layer weights', len(weights1), weights1.shape)
# 	print('Shape of first layer bias', len(bias1), bias1.shape)
# 	for idx in range(weights1.shape[1]):
# 		weights_zero = np.extract(weights1[:,idx]<0.0001, weights1[:,idx])
# 		print('There are %d zero entries in weights' %len(weights_zero))
# 	print('First layer bias', bias1)
# 	#print(predictor.get_layer(index=2).get_weights())
# 	weights2, bias2 = predictor.get_layer(index=2).get_weights()
# 	for idx in range(weights2.shape[1]):
# 		print(weights2[:,idx])
# 		weights_zero = np.extract(weights2[:,idx]<0.0001, weights2[:,idx])
# 		print('There are %d zero entries in weights' %len(weights_zero))	
# 	print('Second layer bias', bias2)
	predictor.summary()
	### examine intermediate activations
	
	for idx in range(10):
		print('True data label', np.argmax(labels_tt[idx]))
		actv = np.matmul(data_tt[idx],weights1)+bias1
		#actv[actv<0] = 0
		print('First layer output before activation at test', actv)
		for idx in range(len(actv)):
			if actv[idx] < 0:
				actv[idx] = 0
		print('First layer activation output at test', actv)
		output = np.matmul(np.transpose(weights2),actv)+bias2
		print('Output before softmax', output)
	
	