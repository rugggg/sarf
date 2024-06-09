import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import random
# Seeding for random operations
main_rng = random.PRNGKey(42)


## PyTorch Data Loading
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10

class WaveSegEncoder(nn.Module):

    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # conv down
        x = nn.Conv(features=8, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))
 
        x = nn.Conv(features=16, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        # base
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = nn.Dense(features=self.latent_dim)(x)
        # 
        return x


class WaveSegDecoder(nn.Module):

    latent_dim: int
    c_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*16*3)(x)
        x = nn.relu(x)
        
        x = x.reshape(x.shape[0], 4, 4, -1)
 
        x = nn.ConvTranspose(features=16, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        # base
        x = nn.ConvTranspose(features=64, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=3, kernel_size=(3,3), strides=(2,2))(x)
        x = nn.tanh(x)

        return x

def mse_recon_loss(model, params, batch):
    imgs, _ = batch
    recon_imgs = model.apply({'params': params}, imgs)
    loss = ((recon_imgs - imgs) ** 2).mean(axis=0).sum()  # Mean over batch, sum over pixels
    return loss


class WaveSegSegment(nn.Module):
    def setup(self):
        self.encoder = WaveSegEncoder(latent_dim=128)
        self.decoder = WaveSegDecoder(latent_dim=128)


def encoder_test():
    ## Test encoder implementation
    # Random key for initialization
    rng = random.PRNGKey(0)
    # Example images as input
    imgs = next(iter(train_loader))[0]
    # Create encoder
    encoder = WaveSegEncoder(latent_dim=128)
    # Initialize parameters of encoder with random key and images
    params = encoder.init(rng, imgs)['params']
    # Apply encoder with parameters on the images
    out = encoder.apply({'params': params}, imgs)
    out.shape
    print(out)
    print(out.shape)
    del out, encoder, params

def decoder_test():
    ## Test decoder implementation
    # Random key for initialization
    rng = random.PRNGKey(0)
    # Example latents as input
    rng, lat_rng = random.split(rng)
    latents = random.normal(lat_rng, (16, 128))
    # Create decoder
    decoder = WaveSegDecoder(latent_dim=128, c_out=3)
    # Initialize parameters of decoder with random key and latents
    rng, init_rng = random.split(rng)
    params = decoder.init(init_rng, latents)['params']
    # Apply decoder with parameters on the images
    out = decoder.apply({'params': params}, latents)
    out.shape

    del out, decoder, params




# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255. * 2. - 1.
    return img

# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs
# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

if __name__ == "__main__":

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root='./cifar/', train=True, transform=image_to_numpy, download=True)
    train_set, val_set = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))


    # data
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate, persistent_workers=True)


    encoder_test()
    decoder_test()
