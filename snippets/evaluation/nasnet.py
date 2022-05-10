from toolz import curry

import tensorflow as tf

from tensorflow.keras.metrics import CategoricalAccuracy, Mean, CategoricalCrossentropy

from hanser.distribute import setup_runtime, distribute_datasets
from hanser.datasets.classification.cifar import make_cifar10_dataset
from hanser.transform import random_crop, cutout, normalize, to_tensor

from hanser.train.optimizers import SGD
from hanser.train.callbacks import DropPathRateScheduleV2
from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import Genotype
from hanser.train.cls import SuperLearner
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy

@curry
def transform(image, label, training):

    if training:
        image = random_crop(image, (32, 32), (4, 4))
        image = tf.image.random_flip_left_right(image)

    image, label = to_tensor(image, label)
    image = normalize(image, [0.491, 0.482, 0.447], [0.247, 0.243, 0.262])

    if training:
        image = cutout(image, 16)

    label = tf.one_hot(label, 10)

    return image, label

mul = 1
batch_size = 96 * mul
eval_batch_size = batch_size
ds_train, ds_test, steps_per_epoch, test_steps = make_cifar10_dataset(
    batch_size, eval_batch_size, transform, drop_remainder=True)

setup_runtime(fp16=True)
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

PC_DARTS_cifar = Genotype(
    normal=[
        ('sep_conv_3x3', 1), ('skip_connect', 0),
        ('sep_conv_3x3', 0), ('dil_conv_3x3', 1),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_5x5', 1), ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)
    ],
    reduce_concat=[2, 3, 4, 5]
)

drop_path = 0.3
model = NASNet(36, 20, True, drop_path, 10, PC_DARTS_cifar)
model.build((None, 32, 32, 3))
model.summary()

criterion = CrossEntropy(auxiliary_weight=0.4)

base_lr = 0.025
epochs = 600
lr_schedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs, min_lr=0)
optimizer = SGD(lr_schedule, momentum=0.9, weight_decay=3e-4, nesterov=True)

train_metrics = {
    'loss': Mean(),
    'acc': CategoricalAccuracy(),
}
eval_metrics = {
    'loss': CategoricalCrossentropy(from_logits=True),
    'acc': CategoricalAccuracy(),
}

learner = SuperLearner(
    model, criterion, optimizer, grad_clip_norm=5.0,
    train_metrics=train_metrics, eval_metrics=eval_metrics,
    work_dir="./CIFAR10-PC-DARTS")

learner.fit(ds_train, epochs, ds_test, val_freq=1,
            steps_per_epoch=steps_per_epoch, val_steps=test_steps,
            callbacks=[DropPathRateScheduleV2()])