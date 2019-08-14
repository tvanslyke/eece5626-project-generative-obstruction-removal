import tensorflow as tf
import numpy as np
from functools import partial
import tqdm
from datetime import datetime
from PIL import Image
import os
import sockserv
import threading
from helpers import WithVariableScope, add_obstruction, repoint_latest_path

# def conv2d(batch, filters, strides, dilations=(1, 1, 1, 1), name=None):
# 	return tf.nn.conv2d(
# 		batch,
# 		filters=filters,
# 		strides=strides,
# 		padding="SAME",
# 		use_cudnn_on_gpu=True,
# 		data_format='NHWC',
# 		dilations=dilations,
# 		name=None
# 	)

def conv2d_transpose(batch, filters, strides, dilations=(1, 1, 1, 1), name=None):
	return tf.nn.conv2d_transpose(
		batch,
		filters=filters,
		output_shape = (batch.shape[0], batch.shape[1] * strides, batch.shape[2] * strides, filters.shape[2]),
		strides=strides,
		padding="SAME",
		data_format='NHWC',
		dilations=dilations,
		name=name,
	)	

def gen_first_layer(batch, weights, bias):
	return tf.reshape(
		tf.nn.leaky_relu(tf.nn.bias_add(tf.squeeze(tf.linalg.matmul(weights, batch)), bias), alpha=0.2),
		(-1, 2, 2, 512)
	)
	

def apply_conv_layers(batch, weights, biases, strides):
	last_layer = batch
	layer = None
	for w, b, s in zip(weights, biases, strides):
		layer = conv2d_transpose(last_layer, w, s)
		layer = tf.nn.bias_add(layer, b)
		last_layer = tf.nn.leaky_relu(layer, alpha=0.2)
	del last_layer
	assert layer is not None
	return layer


xav_init = tf.contrib.layers.xavier_initializer()

def make_vae_encoder(trainable=True):
	layers = (
		tf.layers.Conv2D(
			filters=32,
	 		kernel_size=(5, 5),
	 	       	strides=(2, 2),
	 	       	padding="same",
	 	       	dilation_rate=1,
	 	       	activation=tf.nn.leaky_relu,
	 	       	use_bias=True,
	 	       	kernel_initializer=xav_init,
	 	       	trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_1",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2D(
			filters=64,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_2",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2D(
			filters=128,
			kernel_size=(2, 2),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_3",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2D(
			filters=256,
			kernel_size=(2, 2),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_4",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Flatten(),
		tf.layers.Dense(
			units=128,
			activation=None,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
	)
	@WithVariableScope("vae_encoder", reuse=tf.AUTO_REUSE)
	def encoder_fn(x):
		for layer in layers:
			x = layer(x)
		return x
	return encoder_fn

def make_vae_decoder(trainable=True):
	layers = (
		tf.layers.Dense(
			units=256 * 2 * 2,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.reshape(x, (-1, 2, 2, 256)),
		tf.layers.Conv2DTranspose(
			filters=128,
			kernel_size=(2, 2),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_1",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2DTranspose(
			filters=64,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_2",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2DTranspose(
			filters=32,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.contrib.layers.layer_norm(
			x,
			trainable=trainable,
			scope="layer_norm_3",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2DTranspose(
			filters=3,
			kernel_size=(5, 5),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=None,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		)
	)
	@WithVariableScope("vae_decoder", reuse=tf.AUTO_REUSE)
	def decoder_fn(x):
		for layer in layers:
			x = layer(x)
		return tf.squeeze(x)
	return decoder_fn
	

def vae_loss(inputs, outputs):
	return tf.reduce_mean(tf.math.squared_difference(inputs, outputs))

image_shape = (32, 32, 3)
batch_size = 32
noise_size = 64
encoding_size = 128
rng_seed = 0xdeadbeef
learning_rate = 1e-4



if __name__ == '__main__':
	import cifar
	from matplotlib import pyplot as plt
	encoder = make_vae_encoder()
	decoder = make_vae_decoder()
	batches = (
		cifar.get_batch("data_batch_1")[0],
		cifar.get_batch("data_batch_2")[0],
		cifar.get_batch("data_batch_3")[0],
		cifar.get_batch("data_batch_4")[0],
		cifar.get_batch("data_batch_5")[0]
	)
	data = np.float32(np.concatenate(batches, axis=0))
	print("data.shape={}".format(data.shape))
	
	
	data_mean =  tf.constant(np.mean(data, axis=0))
	data_stdev = tf.constant(np.std(data, axis=0))
	
	standardize = lambda x: (x - data_mean) / data_stdev
	destandardize = lambda x: x * data_stdev + data_mean

	dataset = tf.constant(data)

	inds = tf.random.uniform(
		minval=0,
		maxval=data.shape[0],
		shape=[batch_size],
		dtype=tf.dtypes.int32,
	)
	batch = tf.gather(dataset, inds)
	
	# add obstructed examples to the batch
	obstruction_locs = tf.random.uniform(minval=0, maxval=32, shape=[batch_size // 2, 2], dtype=tf.int32)
	radii = tf.random.uniform(minval=3, maxval=8, shape=[batch_size // 2], dtype=tf.int32)
	obstructed_batch = add_obstruction(batch[batch_size // 2:], obstruction_locs, radii)
	batch = tf.concat([batch[:batch_size // 2], obstructed_batch], axis=0)
	batch = standardize(batch)
	
	encoded = encoder(batch)
	decoded = decoder(encoded)
	
	loss = vae_loss(batch, decoded)
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=tf.trainable_variables())

	encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_encoder')
	decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_decoder')

	saver = tf.train.Saver(var_list=[*encoder_vars, *decoder_vars])
	
	train_iters = 1000000
	
	now = datetime.now()
	output_path = "output/vae/vae-{}".format(now.isoformat())
	save_path = "save/vae/vae-{}".format(now.isoformat())
	repoint_latest_path(output_path, "output/vae/latest")
	repoint_latest_path(save_path, "save/vae/latest")
	
	display_image_index = tf.random.uniform(minval=0, maxval=batch_size, shape=[], dtype=tf.int32)
	input_image = destandardize(batch[display_image_index])
	output_image = destandardize(decoded[display_image_index])

	loss_history = []
	thrd = threading.Thread(target=sockserv.cmdline, args=({"l": loss_history},))
	thrd.start()
	
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.allow_growth=True
	i = 0
	try:
		with tf.Session(config=config) as sess:
			sess.run(tf.variables_initializer(tf.global_variables()))
			print("output path='{}'".format(output_path))
			print("save path='{}'".format(save_path))
			for i in tqdm.trange(train_iters):
				sess.run(opt)
				if (i % 1000) == 0:
					loss_v = sess.run(loss)
					loss_history.append(loss_v)
					print("loss = {}".format(loss_v))
				if (i % 10000) == 0:
					in_img, out_img = sess.run((input_image, output_image))
					in_img = np.uint8(np.clip(in_img, 0, 255))
					out_img = np.uint8(np.clip(out_img, 0, 255))
					save_img = np.hstack((in_img, out_img))
					Image.fromarray(save_img).save("{}/{:04d}.png".format(output_path, i // 10000))
				if (i % 100000) == 0:
					saver.save(sess, save_path, global_step=i)
	finally:
		print(loss_history)
		saver.save(sess, save_path, global_step=train_iters)
		with open("{}/loss_history.txt".format(output_path), "w") as f:
			f.write(repr(loss_history))
	plt.figure()
	plt.plot(loss_history)
	plt.savefig("{}/vae.png".format(output_path))








