import numpy as np

import tensorflow as tf
from tensorflow.keras.initializers import Constant

from tfnas.models.resnetpp.search import Network


class NetworkExt(Network):

    def _initialize_alphas(self):
        k = sum(4 + i for i in range(self.splits))
        total_layers = sum([
            len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)])
        # Start block is not searchable
        self.alphas = self.add_weight(
            'alphas', (total_layers - 4, k), initializer=Constant(0.), trainable=True,
        )

    def call(self, x):
        alphas = tf.nn.sigmoid(self.alphas)

        x = self.stem(x)
        alphas = tf.cast(alphas, x.dtype)

        i = 0
        for l in self.layer1:
            x = l(x, alphas[i])
            if not l.start_block:
                i += 1
        for l in self.layer2:
            x = l(x, alphas[i])
            if not l.start_block:
                i += 1
        for l in self.layer3:
            x = l(x, alphas[i])
            if not l.start_block:
                i += 1
        for l in self.layer4:
            x = l(x, alphas[i])
            if not l.start_block:
                i += 1
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def genotype(self, threshold=0.9):
        alphas = tf.convert_to_tensor(self.alphas.numpy())
        alphas = tf.nn.sigmoid(alphas).numpy()

        layer_cumsum = np.cumsum([
            0, len(self.layer1) - 1, len(self.layer2) - 1, len(self.layer3) - 1])

        genotype = []
        for s in range(alphas.shape[0]):
            if s in layer_cumsum:
                genotype.append(None)
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
    return NetworkExt(base_width=26, scale=4, se_last=True, se_reduction=(4, 8, 8, 8), se_mode=0)