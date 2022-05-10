import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

from hanser.models.layers import Conv2d, Norm, Act, Linear, NormAct, Identity, GlobalAvgPool, Dropout
from hanser.models.modules import DropPath, AntiAliasing
from hanser.models.attention import SELayer
from hanser.models.common.res2net.layers import Res2Conv
from hanser.models.common.modules import get_shortcut_vd
from hanser.models.imagenet.iresnet.resnet import _make_layer, _get_kwargs
from hanser.models.imagenet.stem import SpaceToDepthStem


class PPConv(Layer):

    def __init__(self, channels, splits):
        super().__init__()
        self.splits = splits
        C = channels // splits

        self.ops = []
        for i in range(self.splits):
            if i == 0:
                op = Identity()
            else:
                op = Conv2d(C, C, 3, 1, norm='def', act='def')
            self.ops.append(op)

    def call(self, x, alphas):
        states = list(tf.split(x, self.splits, axis=-1))
        offset = 0
        for i in range(self.splits):
            alphas_i = alphas[offset:offset + len(states)]
            alphas_i = alphas_i / tf.reduce_sum(alphas_i)
            x = sum(alphas_i[j] * h for j, h in enumerate(states))
            x = self.ops[i](x)
            offset += len(states)
            states.append(x)

        return tf.concat(states[-self.splits:], axis=-1)


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, drop_path=0,
                 start_block=False, end_block=False, exclude_bn0=False,
                 se_reduction=4, se_mode=0, se_last=False,
                 base_width=26, scale=4):
        super().__init__()
        self.start_block = start_block
        self.se_last = se_last

        out_channels = channels * self.expansion
        channels = math.floor(channels * (base_width / 64)) * scale

        if not start_block:
            if exclude_bn0:
                self.act0 = Act()
            else:
                self.norm_act0 = NormAct(in_channels)

        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')

        if start_block:
            self.conv2 = Res2Conv(channels, channels, kernel_size=3, stride=1,
                             scale=scale, start_block=True, norm='def', act='def')
        else:
            self.conv2 = PPConv(channels, scale)
        self.aa = AntiAliasing(kernel_size=3, stride=2) if stride == 2 else None

        if not self.se_last:
            self.se = SELayer(channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if start_block:
            self.bn3 = Norm(out_channels)

        if self.se_last:
            self.se = SELayer(out_channels, se_channels=out_channels // se_reduction, mode=se_mode)

        self.drop_path = DropPath(drop_path) if drop_path else Identity()

        if end_block:
            self.norm_act3 = NormAct(out_channels)

        self.shortcut = get_shortcut_vd(in_channels, out_channels, stride)

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def call(self, x, alphas):
        identity = self.shortcut(x)

        if not self.start_block:
            if self.exclude_bn0:
                x = self.act0(x)
            else:
                x = self.norm_act0(x)

        x = self.conv1(x)

        if self.start_block:
            x = self.conv2(x)
        else:
            x = self.conv2(x, alphas)

        if self.aa is not None:
            x = self.aa(x)

        if not self.se_last:
            x = self.se(x)

        x = self.conv3(x)

        if self.start_block:
            x = self.bn3(x)

        if self.se_last:
            x = self.se(x)

        x = self.drop_path(x)
        x = x + identity

        if self.end_block:
            x = self.norm_act3(x)
        return x


class Network(Model):

    def __init__(self, layers=(3, 4, 8, 3), num_classes=1000, channels=(64, 64, 128, 256, 512),
                 strides=(1, 2, 2, 2), dropout=0, **kwargs):
        super().__init__()
        self.splits = kwargs.get("scale", 4)
        block = Bottleneck

        stem_channels, *channels = channels
        self.stem = SpaceToDepthStem(stem_channels)
        c_in = self.stem.out_channels

        blocks = (block,) * 4 if not isinstance(block, tuple) else block
        for i, (block, c, n, s) in enumerate(zip(blocks, channels, layers, strides)):
            layer = _make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i), return_seq=False)
            c_in = c * block.expansion
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

        self._initialize_alphas()

        self.fair_loss_weight = self.add_weight(
            name="fair_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(1.),
            trainable=False,
        )

        self.conn_loss_weight = self.add_weight(
            name="conn_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(1.),
            trainable=False,
        )

        self.l2_loss_weight = self.add_weight(
            name="l2_loss_weight", shape=(),
            dtype=self.dtype, initializer=Constant(1.),
            trainable=False,
        )

    def param_splits(self):
        return slice(None, -4), slice(-4, None)

    def _initialize_alphas(self):
        k = sum(4 + i for i in range(self.splits))

        self.alphas = self.add_weight(
            'alphas', (4, k), initializer=Constant(0.), trainable=True,
        )

    def call(self, x):
        alphas = tf.nn.sigmoid(self.alphas)

        x = self.stem(x)
        alphas = tf.cast(alphas, x.dtype)

        for l in self.layer1:
            x = l(x, alphas[0])
        for l in self.layer2:
            x = l(x, alphas[1])
        for l in self.layer3:
            x = l(x, alphas[2])
        for l in self.layer4:
            x = l(x, alphas[3])
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def arch_loss(self):
        probs = tf.nn.sigmoid(self.alphas)
        fair_loss = -tf.square((probs - 0.5))
        fair_loss = tf.reduce_mean(fair_loss)
        fair_loss = fair_loss + 0.25

        k = self.splits
        n = tf.range(k)
        indices = k * n + n * (n - 1) // 2
        indices = n[:, None] + indices[None, :]
        weights = tf.gather(probs, indices, axis=1)
        weight_sum = tf.reduce_sum(weights, axis=2)
        conn_loss = tf.where(
            weight_sum > 1.,
            tf.zeros_like(weight_sum),
            tf.square(weight_sum - 1),
        )
        conn_loss = tf.reduce_mean(conn_loss)

        l2_loss = 0.5 * tf.square(self.alphas)
        l2_loss = tf.reduce_mean(l2_loss)
        return self.fair_loss_weight * fair_loss + self.conn_loss_weight * conn_loss + self.l2_loss_weight * l2_loss

    def genotype(self, threshold=0.9):
        alphas = tf.convert_to_tensor(self.alphas.numpy())
        alphas = tf.nn.sigmoid(alphas).numpy()

        genotype = []
        for s in range(alphas.shape[0]):
            conn = []
            offset = 0
            for i in range(self.splits):
                a = alphas[s, offset:offset + i + self.splits]
                cs = np.arange(len(a))[a > threshold] + 1
                conn.append(tuple(cs))
                offset += self.splits + i
            genotype.append(conn)
        return tuple(genotype)


def resnet_m():
    return Network(base_width=26, scale=4, se_last=True, se_reduction=(4, 8, 8, 8), se_mode=0)