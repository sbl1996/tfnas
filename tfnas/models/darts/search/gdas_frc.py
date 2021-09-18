import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Layer

from hanser.models.layers import Norm, Conv2d, GlobalAvgPool, Linear, Act, Pool2d
from hanser.ops import gumbel_softmax

from tfnas.models.darts.operations import ReLUConvBN
from tfnas.models.darts.genotypes import get_primitives, Genotype
from tfnas.models.darts.search.gdas import Cell


class NormalCell(Cell):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction_prev, drop_path):
        super().__init__(
            steps, multiplier, C_prev_prev, C_prev, C, False, reduction_prev, drop_path)


class ReductionCell(Layer):

    def __init__(self, C_prev_prev, C_prev, C):
        super().__init__()
        self.reduction = True

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1)

        self.branch_a1 = Sequential([
            Act(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            Norm(C, affine=True),
            Act(),
            Conv2d(C, C, 1),
            Norm(C, affine=True),
        ])
        self.branch_a2 = Sequential([
            Pool2d(3, stride=2, type='max'),
            Norm(C, affine=True),
        ])
        self.branch_b1 = Sequential([
            Act(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            Norm(C, affine=True),
            Act(),
            Conv2d(C, C, 1),
            Norm(C, affine=True),
        ])
        self.branch_b2 = Sequential([
            Pool2d(3, stride=2, type='max'),
            Norm(C, affine=True),
        ])

    def call(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        x0 = self.branch_a1(s0)
        x1 = self.branch_a2(s1)

        x2 = self.branch_b1(s0)
        x3 = self.branch_b2(s1)

        return tf.concat([x0, x1, x2, x3], axis=-1)


class Network(Model):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, drop_path=0, num_classes=10):
        super().__init__()
        self._C = C
        self._steps = steps
        self._multiplier = multiplier
        self._drop_path = drop_path

        C_curr = stem_multiplier * C
        self.stem = Sequential([
            Conv2d(3, C_curr, 3, bias=False),
            Norm(C_curr, affine=True),
        ])

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if reduction:
                cell = ReductionCell(C_prev_prev, C_prev, C_curr)
            else:
                cell = NormalCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev, drop_path)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = GlobalAvgPool()
        self.classifier = Linear(C_prev, num_classes)

        k = sum(2 + i for i in range(self._steps))
        num_ops = len(get_primitives())
        self.alphas_normal = self.add_weight(
            'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
            experimental_autocast=False)

        self.tau = self.add_weight(
            'tau', (), initializer=Constant(1.0), trainable=False, experimental_autocast=False)

    def param_splits(self):
        return slice(None, -1), slice(-1, None)

    def call(self, x):
        s0 = s1 = self.stem(x)
        hardwts_normal = gumbel_softmax(self.alphas_normal, self.tau, hard=True)
        for cell in self.cells:
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                hardwts = tf.cast(hardwts_normal, s0.dtype)
                s0, s1 = s1, cell(s0, s1, hardwts)
        x = self.avg_pool(s1)
        logits = self.classifier(x)
        return logits

    def genotype(self):
        PRIMITIVES = get_primitives()

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return w[i], PRIMITIVES[i]

        def _parse(weights):
            gene = []
            start = 0
            for i in range(self._steps):
                end = start + i + 2
                W = weights[start:end]
                edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
                for j in edges:
                    gene.append((get_op(W[j])[1], j))
                start = end
            return gene

        gene_normal = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype

