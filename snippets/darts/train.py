import math

from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy
from tensorflow_addons.optimizers import AdamW

from hanser.datasets.classification.cifar import load_cifar10
from hanser.datasets.classification.numpy import subsample
from hanser.transform import random_crop, normalize, to_tensor, cutout

from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.datasets import prepare
from hanser.train.optimizers import SGD
from hanser.models.layers import set_defaults

from tfnas.models.darts.search.darts import Network
from tfnas.train.darts import DARTSLearner

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    if training:
        image = cutout(image, 16)

    label = tf.one_hot(label, 10)

    return image, label

batch_size = 64
eval_batch_size = 64

(x_train, y_train), (x_test, y_test) = load_cifar10()

x_train, y_train = subsample(x_train, y_train, ratio=0.01)
x_test, y_test = subsample(x_test, y_test, ratio=0.01)

n_val = len(x_train) // 2
x_val, y_val = x_train[n_val:], y_train[n_val:]
x_train, y_train = x_train[:n_val], y_train[:n_val]

n_train, n_test = len(x_train), len(x_test)
steps_per_epoch = n_train // batch_size
test_steps = math.ceil(n_test / eval_batch_size)

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

ds_train = prepare(ds_train, batch_size, transform(training=True), training=True,
                   buffer_size=n_train, prefetch=False)
ds_val = prepare(ds_val, batch_size, transform(training=False), training=True,
                 buffer_size=n_val, prefetch=False)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False,
                  buffer_size=n_test)
ds_trainval = tf.data.Dataset.zip((ds_train, ds_val))
ds_trainval = ds_trainval.prefetch(tf.data.experimental.AUTOTUNE)

set_defaults({
    'bn': {
        'affine': False,
    }
})

model = Network(4, 5)
model.build((None, 32, 32, 3))

criterion = CrossEntropy()

base_lr = 0.1
epochs = 50
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer_model = SGD(0.025, momentum=0.9, weight_decay=3e-4)
optimizer_arch = AdamW(weight_decay=1e-3, learning_rate=3e-4, beta_1=0.5)


train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = DARTSLearner(
    model, criterion, optimizer_arch, optimizer_model,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./cifar10", grad_clip_norm=5.0)

learner.fit(ds_trainval, epochs, ds_test, val_freq=5,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps)