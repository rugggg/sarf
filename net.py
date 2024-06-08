import jax
import jax.numpy as jnp
import optax
from flax import linen as nn



class WaveSeg(nn.Module):

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
        
        # 
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
      logits = state.apply_fn({'params': params}, images)
      one_hot = jax.nn.one_hot(labels, 10)
      loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
      return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy

model = WaveSeg()
