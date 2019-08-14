# Prerequisites
* Python3.5 with TensorFlow, NumPy, TQDM, and Pillow.
* The CIFAR-10 **binary** dataset should be unpacked to `data/cifar-10/cifar-10-batches-bin/`.

# Running the Model
To train the VAE, run `vae.py` until convergence.
To train the GAN, run `arch.py` until convergence.

GAN model checkpoints will be saved in `save/gan/`, while VAE checkpoints will be saved in `save/vae/`.
`arch.py` will need to be modified so that the hard-coded VAE checkpoint path corresponds to the one on your machine.

GAN training outputs will be saved in `output/gan/`, while VAE checkpoints will be saved in `output/vae/`.

