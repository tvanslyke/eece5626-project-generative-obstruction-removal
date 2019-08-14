import tensorflow as tf
import numpy as np
import os

class WithVariableScope:

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = dict(kwargs)

	def __call__(self, fn):
		def wrapped_fn(*args, **kwargs):
			with tf.variable_scope(*self.args, **self.kwargs):
				return fn(*args, **kwargs)
		return wrapped_fn

def add_obstruction(images, locs, radii):
	x, y = tf.meshgrid(list(range(images.shape[1])), list(range(images.shape[2])))
	x_locs = tf.expand_dims(tf.expand_dims(locs[:, 0], axis=-1), axis=-1)
	y_locs = tf.expand_dims(tf.expand_dims(locs[:, 1], axis=-1), axis=-1)
	radii = tf.expand_dims(tf.expand_dims(radii, axis=-1), axis=-1)
	x = tf.expand_dims(x, 0)
	y = tf.expand_dims(y, 0)
	mask = tf.expand_dims(((x - x_locs) ** 2 + (y - y_locs) ** 2) > (radii ** 2), axis=-1)
	mask = tf.cast(mask, images.dtype)
	return images * mask

def repoint_latest_path(new_path, latest_path):
	new_path, latest_path = os.path.realpath(new_path), os.path.abspath(latest_path)

	print(new_path, latest_path)
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	if os.path.exists(latest_path):
		os.unlink(latest_path)
	os.symlink(new_path, latest_path)


if __name__ == '__main__':
	import cifar
	from matplotlib import pyplot as plt
	images = cifar.get_batch("data_batch_1")[0][:16]
	tf_images = tf.constant(images)
	locs = tf.random.uniform(minval=0, maxval=32, shape=(16, 2), dtype=tf.int32)
	radii = tf.random.uniform(minval=3, maxval=8, shape=(16,), dtype=tf.int32)
	obstructed = add_obstruction(images, locs, radii)
	with tf.Session() as sess:
		obs_v = sess.run(obstructed)
		print(obs_v)
		print(obs_v.dtype)
		obs_v = np.reshape(obs_v, (4, 4, 32, 32, 3))
		images = np.reshape(images, (4, 4, 32, 32, 3))
		fig, axes = plt.subplots(4, 8)
		for i, j in ((x, y) for x in range(4) for y in range(8)):
			if j < 4:
				axes[i][j].imshow(obs_v[i, j])
			else:
				axes[i][j].imshow(images[i, j % 4])
		plt.savefig("save/obstructions.png")
		










