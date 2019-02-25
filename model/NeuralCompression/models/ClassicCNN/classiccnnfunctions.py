import numpy as np
import tensorflow as tf


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
        base_lr=0.1, warmup=False):
    """Get a learning rate that decays step-wise as training progresses.
    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.
      base_lr: Initial learning rate scaled based on batch_denom.
      warmup: Run a 5 epoch warmup to the initial lr.
    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        """Builds scaled learning rate function with 5 epoch warm up."""
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
        if warmup:
            warmup_steps = int(batches_per_epoch * 5)
            warmup_lr = (
                    initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
        return lr

    return learning_rate_fn
