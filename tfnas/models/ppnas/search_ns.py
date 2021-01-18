import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from hanser.models.layers import Conv2d, Norm, Act, Linear, Pool2d, Identity, GlobalAvgPool

from tfnas.models.ppnas.operations import OPS
from tfnas.models.ppnas.genotypes import Genotype


class PPConv(Layer):

    def __init__(self, channels, splits):
        super().__init__()
        self.splits = splits
        C = channels // splits

        self.ops = []
        for i in range(self.splits):
            op = OPS['nor_conv_3x3'](C, 1)
            self.ops.append(op)

    def call(self, x, alphas):
        states = list(tf.split(x, self.splits, axis=-1))
        offset = 0
        for i in range(self.splits):
            x = sum(alphas[offset + j] * h for j, h in enumerate(states))
            x = self.ops[i](x)
            offset += len(states)
            states.append(x)

        return tf.concat(states[-self.splits:], axis=-1)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, base_width, splits):
        super().__init__()
        self.stride = stride

        out_channels = channels * self.expansion
        width = math.floor(out_channels // self.expansion * (base_width / 64)) * splits
        self.conv1 = Conv2d(in_channels, width, kernel_size=1,
                            norm='def', act='def')
        if stride == 1:
            self.conv2 = PPConv(width, splits=splits)
        else:
            self.conv2 = Conv2d(width, width, kernel_size=3, stride=2, groups=splits,
                                norm='def', act='def')
        self.conv3 = Conv2d(width, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros')

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x, alphas):
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.stride == 1:
            x = self.conv2(x, alphas)
        else:
            x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        x = self.act(x)
        return x


def beta_softmax(betas, steps, scale=False):
    beta_list = []
    offset = 0
    for i in range(steps):
        beta = tf.nn.softmax(betas[offset:(offset + i + steps)], axis=0)
        if scale:
            beta = beta * len(beta)
        beta_list.append(beta)
        offset += i + steps
    betas = tf.concat(beta_list, axis=0)
    return betas


class Network(Model):

    def __init__(self, depth=56, base_width=13, splits=4, num_classes=10, stages=(32, 32, 64, 128)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        block = Bottleneck
        layers = [(depth - 2) // 9] * 3
        self.num_stages = len(layers)

        self.stem = Conv2d(3, self.stages[0], kernel_size=3, norm='def', act='def')
        self.in_channels = self.stages[0]

        self.layer1 = self._make_layer(
            block, self.stages[1], layers[0], stride=1,
            base_width=base_width, splits=splits)
        self.layer2 = self._make_layer(
            block, self.stages[2], layers[1], stride=2,
            base_width=base_width, splits=splits)
        self.layer3 = self._make_layer(
            block, self.stages[3], layers[2], stride=2,
            base_width=base_width, splits=splits)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

        self._initialize_alphas()

    def index_model(self, xss):
        return tuple(xs[:-2] for xs in xss)

    def index_arch(self, xss):
        return tuple(xs[-2:] for xs in xss)

    def _initialize_alphas(self):
        k = sum(4 + i for i in range(self.splits))

        self.alphas = self.add_weight(
            'alphas', (self.num_stages, k), initializer=RandomNormal(stddev=1e-3), trainable=True,
        )

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, **kwargs))
        return layers

    def call(self, x):
        alphas = beta_softmax(self.alphas, self.splits)

        x = self.stem(x)
        alphas = tf.cast(alphas, x.dtype)

        for l in self.layer1:
            x = l(x, alphas[0])
        for l in self.layer2:
            x = l(x, alphas[1])
        for l in self.layer3:
            x = l(x, alphas[2])

        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def genotype(self):
        alphas = tf.convert_to_tensor(self.alphas.numpy())
        alphas = beta_softmax(alphas, self.splits).numpy()

        offset = 0
        normal = []
        for i in range(self.splits):
            a = alphas[offset:offset + i + self.splits]
            c1, c2 = np.argpartition(-a, 2)[:2] + 1
            normal.append((c1, c2, 'nor_conv_3x3'))
            offset += self.splits + i
        return Genotype(normal=normal)
