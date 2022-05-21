from toolz import curry

import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.transform import random_crop, normalize, to_tensor

from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy
from hanser.train.optimizers import SGD, AdamW
from hanser.models.layers import set_defaults

from tfnas.train.darts import DARTSLearner
from tfnas.models.darts.search.gdas import Network
from tfnas.models.nasnet.primitives import set_primitives
from tfnas.datasets.cifar import make_darts_cifar10_dataset
from tfnas.train.callbacks import PrintGenotype, TauSchedule

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)
        # image = autoaugment(image, "CIFAR10")

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    # if training:
    #     image = cutout(image, 16)

    label = tf.one_hot(label, 10)

    return image, label

batch_size = 64
eval_batch_size = 64

ds_train, ds_eval, steps_per_epoch, eval_steps = make_darts_cifar10_dataset(
    batch_size, eval_batch_size, transform, drop_remainder=True)

setup_runtime(fp16=True)
ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

set_defaults({
    'bn': {
        'affine': False,
        'track_running_stats': False,
    },
})

set_primitives('darts')

model = Network(16, 8)
model.build((None, 32, 32, 3))

criterion = CrossEntropy()

base_lr = 0.025
epochs = 240
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=1e-3)
optimizer_model = SGD(lr_schedule, momentum=0.9, weight_decay=3e-4)
optimizer_arch = AdamW(learning_rate=3e-4, beta_1=0.5, weight_decay=1e-3)


train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = DARTSLearner(
    model, criterion, optimizer_arch, optimizer_model, steps_per_loop=1,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models/darts_search", grad_clip_norm=5.0)

learner.fit(ds_train, epochs, ds_eval, val_freq=5,
            steps_per_epoch=steps_per_epoch, val_steps=eval_steps,
            callbacks=[PrintGenotype(1), TauSchedule(tau_max=10.0, tau_min=0.1)])