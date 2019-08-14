import tensorflow as tf
import numpy as np
import vae
import os
import tqdm
import threading
from datetime import datetime
from PIL import Image
from vae import xav_init
from helpers import WithVariableScope, repoint_latest_path, add_obstruction
import cifar
import sockserv


def make_generator(trainable=True):
	layers = (
		tf.layers.Dense(
			units=1024 * 2 * 2,
			activation=None,
			use_bias=False,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.reshape(x, (-1, 2, 2, 1024)),
		tf.layers.Conv2DTranspose(
			filters=512,
			kernel_size=(3, 3),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=False,
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
			filters=256,
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
			filters=128,
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
			scope="layer_norm_4",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2DTranspose(
			filters=3,
			kernel_size=(5, 5),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.sigmoid,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		)
	)
	@WithVariableScope("generator", reuse=tf.AUTO_REUSE)
	def generator_fn(batch):
		for layer in layers:
			batch = layer(batch)
		return batch
	return generator_fn

def make_discriminator(trainable=True):
	layers = (
		tf.layers.Conv2D(
			filters=128,
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
			filters=256,
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
			scope="layer_norm_2",
			reuse=tf.AUTO_REUSE
		),
		tf.layers.Conv2D(
			filters=512,
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
		tf.layers.Conv2D(
			filters=1024,
			kernel_size=(2, 2),
			strides=(2, 2),
			padding="same",
			dilation_rate=1,
			activation=tf.nn.leaky_relu,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		lambda x: tf.reshape(x, shape=[x.shape[0], -1]),
		tf.layers.Dense(
			units=64,
			activation=tf.nn.leaky_relu,
			use_bias=False,
			kernel_initializer=xav_init,
			trainable=trainable
		),
		tf.layers.Dense(
			units=1,
			activation=None,
			use_bias=True,
			kernel_initializer=xav_init,
			trainable=trainable
		),
	)
	@WithVariableScope("discriminator", reuse=tf.AUTO_REUSE)
	def discriminator_fn(batch):
		for layer in layers:
			batch = layer(batch)
		return batch
	return discriminator_fn

def make_gradient_penalty(discrim, real, fake, lam = 10, return_norm=True):
	eps = tf.random_uniform(
		shape=[int(real.shape[0]), *[1 for _ in real.shape[1:]]],
		minval=0.0,
		maxval=1.0,
		dtype=tf.float32
	)
	interp = real + eps * (fake - real)
	discrimmed = discrim(interp)
	grad = tf.gradients(discrimmed, [interp])[0]
	grad_norm = tf.norm(grad, axis=-1)
	penalty = lam * tf.reduce_mean(tf.square(tf.math.maximum(0.0, grad_norm - 1)))
	if return_norm:
		return penalty, grad_norm
	else:
		return penalty

learning_rate = 1e-4
batch_size = 32
noise_size = encoding_size = 128
rng_seed = 0xdeadbeef


if __name__ == '__main__':
	gen_ = make_generator()
	gen = lambda x: standardize(255.0 * gen_(x))
	discrim = make_discriminator()
	encoder = vae.make_vae_encoder(trainable=False)
	decoder = vae.make_vae_decoder(trainable=False)

	batches = (
		cifar.get_batch("data_batch_1")[0],
		cifar.get_batch("data_batch_2")[0],
		cifar.get_batch("data_batch_3")[0],
		cifar.get_batch("data_batch_4")[0],
		cifar.get_batch("data_batch_5")[0]
	)
	data = np.float32(np.concatenate(batches, axis=0))

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

	obstructed_batch = add_obstruction(
		batch,
		tf.random.uniform(minval=0, maxval=32, shape=[batch_size, 2], dtype=tf.int32),
		tf.random.uniform(minval=3, maxval=8, shape=[batch_size], dtype=tf.int32)
	)

	standardized_batch = standardize(batch)
	standardized_obstructed_batch = standardize(obstructed_batch)

	unobstructed_encs = encoder(standardized_batch)
	obstructed_encs = encoder(standardized_obstructed_batch)

	noise = tf.random_uniform((batch_size, noise_size), minval=-1.0, maxval=1.0, dtype=tf.float32, seed=rng_seed)
	
	gen_input = tf.reshape(tf.concat((unobstructed_encs, noise), axis=-1), (batch_size, noise_size + encoding_size))

	generated = gen(gen_input)
	destandardized_gen = destandardize(generated)
	generated_encs = encoder(generated)

	discrim_true_input = tf.concat((standardized_obstructed_batch, standardized_batch), axis=-1)
	discrim_fake_input = tf.concat((standardized_obstructed_batch, generated), axis=-1)

	d_real_logits, d_fake_logits = discrim(discrim_true_input), discrim(discrim_fake_input)
	real_logits_mean, fake_logits_mean = tf.reduce_mean(d_real_logits), tf.reduce_mean(d_fake_logits)

	grad_penalty = make_gradient_penalty(discrim, discrim_true_input, discrim_fake_input, lam=10, return_norm=False)

	g_loss = -fake_logits_mean
	d_loss = (fake_logits_mean - real_logits_mean) + grad_penalty

	print(fake_logits_mean)
	print(real_logits_mean)
	print(grad_penalty)

	g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

	encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_encoder')
	decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae_decoder')

	g_saver = tf.train.Saver(var_list=g_vars)
	d_saver = tf.train.Saver(var_list=d_vars)
	
	print(g_vars)
	print(d_vars)
	

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		config = dict(learning_rate=learning_rate, beta1=0, beta2=0.9)
		with tf.variable_scope("g_optimizer"):
			g_opt = tf.train.AdamOptimizer(**config).minimize(g_loss, var_list=g_vars)
		with tf.variable_scope("d_optimizer"):
			d_opt = tf.train.AdamOptimizer(**config).minimize(d_loss, var_list=d_vars)
	print(g_vars)
	print(d_vars)

	g_loss_history = []
	d_loss_history = []

	thrd = threading.Thread(target=sockserv.cmdline, args=(dict(g=g_loss_history, d=d_loss_history)))
	thrd.start()

	train_iters = 100000
	d_train_rounds = 5

	do_restore = False

	g_restore_path = None if not do_restore else "save/gan/gan-2019-08-10T15:44:43.452134-generator-10000"
	d_restore_path = None if not do_restore else "save/gan/gan-2019-08-10T15:44:43.452134-discriminator-10000"
	
	now = datetime.now()
	output_path = "output/gan/gan-{}".format(now.isoformat())
	save_path = "save/gan/gan-{}".format(now.isoformat())

	repoint_latest_path(output_path, "output/gan/latest")

	loss_print_interval = 10
	img_save_interval = 100
	model_save_interval = 1000
	unobs_save, obs_save, gen_save = batch[:3], obstructed_batch[:3], destandardized_gen[:3]

	try:
		idx = 0
		with tf.Session() as sess:
			# initialize trainable variables (everything not part of the encoder/decoder)
			sess.run(tf.variables_initializer(tf.global_variables()))
			# restore VAE encoder and decoder vars
			tf.train.Saver(var_list=[*encoder_vars, *decoder_vars]).restore(
				sess,
				os.path.realpath("save/vae/vae-2019-08-13T12:46:39.513396-400000")
			)
			print("output path='{}'".format(os.path.realpath(output_path)))
			print("save path='{}'".format(os.path.realpath(save_path)))
			if g_restore_path and d_restore_path:
				g_saver.restore(sess, g_restore_path)
				d_saver.restore(sess, d_restore_path)
			for i in tqdm.trange(train_iters):
				for _ in range(d_train_rounds):
					sess.run(d_opt)
				sess.run(g_opt)
				if (i % loss_print_interval) == 0:
					g_loss_v, d_loss_v = sess.run((g_loss, d_loss))
					g_loss_history.append(g_loss_v)
					d_loss_history.append(d_loss_v)
					print("g_loss = {}, d_loss = {}".format(g_loss_v, d_loss_v))
				if (i % img_save_interval) == 0:
					unobs_v, obs_v, gen_v = sess.run((unobs_save, obs_save, gen_save))
					unobs_v, obs_v, gen_v = (np.uint8(np.clip(im, 0, 255)) for im in (unobs_v, obs_v, gen_v))
					unobs_v = np.concatenate([*unobs_v], axis=0)
					obs_v = np.concatenate([*obs_v], axis=0)
					gen_v = np.concatenate([*gen_v], axis=0)
					save_img = np.concatenate([unobs_v, obs_v, gen_v], axis=1)
					img_name = "{}/{:04d}.png".format(output_path, i // img_save_interval)
					print("saving {}".format(img_name))
					Image.fromarray(save_img).save(img_name)
				if (i % model_save_interval) == 0 and i != 0:
					g_saver.save(sess, save_path="{}-generator".format(save_path), global_step=i)
					d_saver.save(sess, save_path="{}-discriminator".format(save_path), global_step=i)
				idx = i
			g_saver.save(sess, save_path="{}-generator".format(save_path), global_step=train_iters)
			d_saver.save(sess, save_path="{}-discriminator".format(save_path), global_step=train_iters)
	finally:
		print("generator loss history={}".format(g_loss_history))
		print("discriminator loss history={}".format(d_loss_history))
		if g_loss_history or d_loss_history:
			with open("{}_loss_history.txt".format(output_path), "w") as f:
				print("g_loss_history={}".format(g_loss_history), file=f)
				print("d_loss_history={}".format(d_loss_history), file=f)
		g_saver.save(sess, save_path="{}-generator-backup".format(save_path), global_step=idx)
		d_saver.save(sess, save_path="{}-discriminator-backup".format(save_path), global_step=idx)









