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
from tfnas.models.darts.search.pc_darts import Network
from tfnas.models.nasnet.primitives import set_primitives
from tfnas.datasets.cifar import make_darts_cifar10_dataset
from tfnas.train.callbacks import PrintGenotype, TrainArch

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

batch_size = 16
eval_batch_size = 16

ds_train, ds_eval, steps_per_epoch, eval_steps = make_darts_cifar10_dataset(
    batch_size, eval_batch_size, transform, drop_remainder=True, sub_ratio=0.001)

# setup_runtime(fp16=True)
# ds_train, ds_eval = distribute_datasets(ds_train, ds_eval)

set_defaults({
    'bn': {
        'affine': False,
        'track_running_stats': False,
    },
})

set_primitives('tiny')

model = Network(8, 5, k=4)
model.build((None, 32, 32, 3))

criterion = CrossEntropy()

base_lr = 0.1
epochs = 50
lr_schedule = CosineLR(base_lr, steps_per_epoch, epochs=epochs, min_lr=1e-3)
optimizer_model = SGD(lr_schedule, momentum=0.9, weight_decay=3e-4)
optimizer_arch = AdamW(learning_rate=6e-4, beta_1=0.5, weight_decay=1e-3)


train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = DARTSLearner(
    model, criterion, optimizer_arch, optimizer_model, jit_compile=False,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir=f"./models/darts_search", grad_clip_norm=5.0)

learner.fit(ds_train, epochs, ds_eval, val_freq=5,
            steps_per_epoch=steps_per_epoch, val_steps=eval_steps,
            callbacks=[PrintGenotype(16), TrainArch(16)])