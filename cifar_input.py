# -*-coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os



# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


tf.app.flags.DEFINE_string('data_dir','/home/xhp1/test/scripts/data/cifar10','directory of cifar10 dataset')
BATCH_SIZE=100


def _sparse_to_dense(labels):
	'''Convert sparese labels to one hot format.
	Args:
		labels:[1,2,1,1] like this
	Retruns:
		dense_labels: tensor 
	'''
	#####  make sure to understand sparse_to_dense function ??????????????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#############################

	sparse_labels = tf.reshape(labels, [-1, 1])
	derived_size = tf.shape(sparse_labels)[0]
	indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
	concated = tf.concat(1, [indices, sparse_labels])
	outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(NUM_CLASSES, [1])])
	dense_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
	return dense_labels

def generate_batch(image,label,min_que_example,batch_size):

	
	images,labels=tf.train.shuffle_batch(
		[image,label],
		batch_size=batch_size,
		num_threads=16,
		capacity=min_que_example+3*batch_size,# The maximum number of elements in the queue.
		min_after_dequeue=min_que_example #used to ensure a level of mixing of elements or the minimal number of examples in the queue
		)

	labels_=tf.reshape(labels,[batch_size])
	return images,_sparse_to_dense(labels_)




def read_cifar10(filename_queue):
	''' Reading data from cifry data.
	Args:
		filename_queue: A queue of strings with the filenames to read.
	Returns:
		An cifar object representing a single example,with the following fields as the return of the current return.
		height: number of rows in a single image
		width: number of columns in a single image
		depth: channels of images ,3 means RGB
		key: a scalar string ,tensor describing the filename& record number of this example
		label: an int32 Tensor with the label in the range 0---9
		uint8image: image data with [height,width,depth]  uint8 tensor
	'''
	label_bytes=1
	image_bytes=3072
	record_bytes=3073 # image_bytes+label_bytes
	class cifar(object):
		height=32
		width=32
		depth=3

		def __init__(self,label,key,uint8image):
			self.label=label
			self.key=key
			self.uint8image=uint8image
	reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
	key,value=reader.read(filename_queue) 
	# convert string to a vector of uint8 byte by byte 
	record_bytes=tf.decode_raw(value,tf.uint8)

	label=tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32) # 从[0]下标开始截取，截取[label_bytes]个数据
	uint8image_raw=tf.reshape(tf.slice(record_bytes,[1],[image_bytes]),[cifar.depth,cifar.height,cifar.width])
	uint8image_tranpose=tf.transpose(uint8image_raw,[1,2,0]) # transpose to [height,width,depth] 
	uint8image=tf.cast(uint8image_tranpose,tf.float32)
	return cifar(label,key,uint8image)


def distorted_input(data_dir,batch_size):
	"""Construct distorted input for CIFAR training using the Reader ops.

	Args:
	data_dir: Path to the CIFAR-10 data directory.
	batch_size: Number of images per batch.

	Returns:
	images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	labels: Labels. 1D tensor of [batch_size] size.
	"""
	filenames=[os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]

	for f in filenames:
		if not os.path.exists(f):
			#print "'{}' not exists".formate(f)
			raise ValueError('failed to fined file:'+f)
	filename_queue=tf.train.string_input_producer(filenames)

	original_image=read_cifar10(filename_queue) # return a object of cifar.
	
	# distorted step of image processing 
	distorted_image=tf.random_crop(original_image.uint8image,[IMAGE_SIZE,IMAGE_SIZE,3])# random crop a image to size of 24*24

	'''
		other distorted method to use in future

	'''

	float_image=tf.image.per_image_whitening(distorted_image)

	# Ensure  random shuffle have a good result
	min_que_example=int(0.4*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) #Filling queue with 40% of CIFAR images before starting to train

	return generate_batch(float_image,original_image.label,min_que_example,batch_size)




 


	



def unit_test_input():

	with tf.Session() as sess:
		batch=distorted_input(tf.app.flags.FLAGS.data_dir,batch_size=10)
		print batch[0].eval()
		#sess.run(batch)

		print 'fsdf'	
		# cc=batch
		# print 'begin fetch'
		# cc1=sess.run([batch,cc])
		# print cc1


#unit_test_input()


#distorted_input('C:\Program Files (x86)',10)；