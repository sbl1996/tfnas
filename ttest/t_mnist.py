import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from toolz import curry
from hanser.datasets.mnist import make_mnist_dataset

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras.optimizers import Adam

from hanser.transform import to_tensor, normalize, pad
from hanser.distribute import setup_runtime, distribute_datasets
from hanser.ops import gumbel_softmax
from hanser.train.optimizers import SGD
from hanser.train.lr_schedule import CosineLR

from hanser.models.layers import Conv2d, Linear, Identity, Pool2d, GlobalAvgPool

from tfnas.models.layers import LayerChoice


@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    return image, label


batch_size = 32
eval_batch_size = 64
ds_train, ds_test, steps_per_epoch, test_steps = \
    make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=0.01)

# setup_runtime()
# ds_train, ds_test = distribute_datasets(ds_train, ds_test)

def make_layer_choice(channels):
    return LayerChoice([
        # Identity(),
        Conv2d(channels, channels, kernel_size=1, norm='def', act='def'),
        Conv2d(channels, channels, kernel_size=3, norm='def', act='def'),
        Conv2d(channels, channels, kernel_size=3, groups=channels, norm='def', act='def'),
        Conv2d(channels, channels, kernel_size=5, norm='def', act='def'),
        Conv2d(channels, channels, kernel_size=5, groups=channels, norm='def', act='def'),
    ])


class ConvNet(tf.keras.Model):

    def __init__(self):
        super().__init__()
        channels = 4
        self.stem = Conv2d(1, channels, kernel_size=3, norm='def', act='def')

        self.normal1 = make_layer_choice(channels)
        self.reduce1 = Sequential([
            Pool2d(kernel_size=3, stride=2),
            Conv2d(channels, channels * 2, 1)
        ])
        channels = channels * 2

        self.normal2 = make_layer_choice(channels)
        self.reduce2 = Sequential([
            Pool2d(kernel_size=3, stride=2),
            Conv2d(channels, channels * 2, 1)
        ])
        channels = channels * 2

        self.normal3 = make_layer_choice(channels)

        self.avg_pool = GlobalAvgPool()
        self.fc = Linear(channels, 10)

        self.alpha_normal = self.add_weight(
            name='alpha_normal', shape=(self.normal1.n_choices,), dtype=self.dtype,
            initializer=RandomNormal(stddev=0.01), trainable=True)

        self.tau = self.add_weight(
            name='tau', shape=(), dtype=self.dtype, initializer=Constant(1.0))

    def param_splits(self):
        return slice(None, -1), slice(-1, None)

    def call(self, x):
        hardwts, index = gumbel_softmax(self.alpha_normal, tau=self.tau, hard=True, return_index=True)

        x = self.stem(x)
        x = self.normal1(x, hardwts, index)
        x = self.reduce1(x)
        x = self.normal2(x, hardwts, index)
        x = self.reduce2(x)
        x = self.normal3(x, hardwts, index)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


model = ConvNet()
model.build((None, 32, 32, 1))

epochs = 200
lr_schedule = CosineLR(0.05, steps_per_epoch, epochs, min_lr=0)
optimizer_model = SGD(lr_schedule, momentum=0.9, weight_decay=1e-4, nesterov=True,
                exclude_from_weight_decay=['alpha_normal'])
optimizer_arch = Adam(1e-3)

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function(jit_compile=True)
def train_step(batch):
    x, y = batch
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(y, p, from_logits=True)
        loss = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer_model.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(per_example_loss)
    train_acc.update_state(y, p)


for epoch in range(epochs):

    it = iter(ds_train)
    train_loss.reset_state()
    train_acc.reset_state()
    for step in range(steps_per_epoch):
        batch = next(it)
        train_step(batch)
    print("Epoch %d/%d" % (epoch + 1, epochs))
    print("Train - loss: %.4f, acc: %.4f" % (train_loss.result(), train_acc.result()))
    print(tf.nn.softmax(model.alpha_normal))
