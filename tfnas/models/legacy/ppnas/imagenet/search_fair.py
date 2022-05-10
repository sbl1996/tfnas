import tensorflow as tf
from tensorflow.keras import Sequential
from hanser.models.layers import Conv2d, Linear, Pool2d, GlobalAvgPool

from tfnas.models.ppnas.search_fair import Network as CIFARNetwork, Bottleneck


class Network(CIFARNetwork):

    def __init__(self, layers=(3, 4, 6, 4), base_width=24, splits=4,
                 num_classes=1000, stages=(64, 64, 128, 256, 512)):
        super().__init__()
        self.stages = stages
        self.splits = splits
        block = Bottleneck
        self.num_stages = len(layers)

        self.stem = Sequential([
            Conv2d(3, self.stages[0] // 2, kernel_size=3, stride=2,
                   norm='def', act='def'),
            Conv2d(self.stages[0] // 2, self.stages[0] // 2, kernel_size=3,
                   norm='def', act='def'),
            Conv2d(self.stages[0] // 2, self.stages[0], kernel_size=3,
                   norm='def', act='def'),
        ])
        self.maxpool = Pool2d(kernel_size=3, stride=2, type='max')
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
        self.layer4 = self._make_layer(
            block, self.stages[4], layers[3], stride=2,
            base_width=base_width, splits=splits)

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.in_channels, num_classes)

        self._initialize_alphas()

    def _make_layer(self, block, channels, blocks, stride, **kwargs):
        layers = [block(self.in_channels, channels, stride=stride, **kwargs)]
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels, stride=1, **kwargs))
        return layers

    def call(self, x):
        alphas = tf.nn.sigmoid(self.alphas)

        x = self.stem(x)
        x = self.maxpool(x)
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
        x = self.fc(x)
        return x