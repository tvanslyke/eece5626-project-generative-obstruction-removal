import pickle
import numpy as np


def label_names():
	with open("data/cifar-10/cifar-10-batches-bin/batches.meta.txt", "rt") as f:
		return tuple((line.strip() for line in f.read().split("\n") if line.strip()))

def read_batch(f):
	images = np.zeros(dtype=np.uint8, shape=(10000, 32, 32, 3))
	labels = np.zeros(dtype=np.uint8, shape=(10000,))
	for i in range(10000):
		labels[i] = ord(f.read(1))
		img = images[i]
		img[..., 0] = np.fromfile(f, dtype=np.uint8, count=1024).reshape((32, 32))
		img[..., 1] = np.fromfile(f, dtype=np.uint8, count=1024).reshape((32, 32))
		img[..., 2] = np.fromfile(f, dtype=np.uint8, count=1024).reshape((32, 32))
	return images, labels

def get_batch(name):
	with open("data/cifar-10/cifar-10-batches-bin/{}.bin".format(name), "rb") as f:
		return read_batch(f)

def by_label(images, labels):
	labelled = tuple(([] for _ in range(10)))
	for image, label in zip(images, labels):
		labelled[label].append(image)
	return dict(zip(label_names(), (np.asarray(l) for l in labelled)))

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	images, labels = get_batch("test_batch")
	print(images.shape)
	# for i in range(10):
	# 	img = images[i + 3]
	# 	print(img.shape)
	# 	label = label_names()[labels[i + 3]]
	# 	plt.figure()
	# 	plt.imshow(img)
	# 	plt.title(label)
	# 	plt.show()

	

