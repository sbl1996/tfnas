import math

import tensorflow as tf

from hanser.datasets import prepare
from hanser.datasets.classification.cifar import load_cifar10, load_cifar100
from hanser.datasets.classification.numpy import subsample


def make_darts_cifar10_dataset(
    batch_size, eval_batch_size, transform, drop_remainder=None, sub_ratio=None):
    return make_cifar_dataset(
        load_cifar10, batch_size, eval_batch_size, transform, drop_remainder, sub_ratio)


def make_darts_cifar100_dataset(
    batch_size, eval_batch_size, transform, drop_remainder=None, sub_ratio=None):
    return make_cifar_dataset(
        load_cifar100, batch_size, eval_batch_size, transform, drop_remainder, sub_ratio)


def make_cifar_dataset(
    load_fn, batch_size, eval_batch_size, transform, drop_remainder=None, sub_ratio=None):

    if drop_remainder is None:
        drop_remainder = False

    (x_train, y_train), (_x_test, _y_test) = load_fn()

    if sub_ratio:
        x_train, y_train = subsample(x_train, y_train, ratio=sub_ratio)

    n_val = len(x_train) // 2
    x_val, y_val = x_train[n_val:], y_train[n_val:]
    x_train, y_train = x_train[:n_val], y_train[:n_val]

    n_train = len(x_train)
    steps_per_epoch = n_train // batch_size
    eval_steps = math.ceil(n_val / eval_batch_size)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    ds_train = prepare(ds_train, batch_size, transform(training=True), training=True,
                       buffer_size=n_train, prefetch=False)
    ds_search = prepare(ds_val, batch_size, transform(training=False), training=True,
                        buffer_size=n_val, prefetch=False)
    ds_train = tf.data.Dataset.zip((ds_train, ds_search))
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = prepare(ds_val, eval_batch_size, transform(training=False), training=False,
                     buffer_size=n_val, drop_remainder=drop_remainder)

    return ds_train, ds_val, steps_per_epoch, eval_steps
