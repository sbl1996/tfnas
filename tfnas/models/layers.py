import tensorflow as tf
from tensorflow.keras.layers import Layer

class LayerChoice(Layer):

    def __init__(self, choices):
        super().__init__()
        self.choices = choices
        self.n_choices = len(self.choices)

    def build(self, input_shape):
        for op in self.choices:
            op.build(input_shape)

    def call(self, x, hardwts, index):
        xs = [
            tf.cond(hardwts[i] == 1, lambda: self.choices[i](x) * hardwts[i], lambda: hardwts[i] * tf.ones_like(x))
            for i in range(self.n_choices)]
        return sum(xs)


# class LayerChoice2(Layer):
#
#     def __init__(self, ops):
#         super().__init__()
#         self.ops = ops
#         self.n_ops = len(self.ops)
#
#         self.tau = self.add_weight(
#             name='tau', shape=(), dtype=self.dtype,
#             initializer=Constant(1.0), trainable=False,
#         )
#
#         self.weight = self.add_weight(
#             name='weight', shape=(self.n_ops,), dtype=self.dtype,
#             initializer=RandomNormal(stddev=1e-2), trainable=True,
#         )
#
#     def _create_branch_fn(self, x, i, hardwts):
#         return lambda: self.ops[i](x) * hardwts[i]
#
#     def build(self, input_shape):
#         for op in self.ops:
#             op.build(input_shape)
#
#     def call(self, x):
#         hardwts, index = gumbel_softmax(self.weight, tau=self.tau, hard=True, return_index=True)
#         branch_fns = [
#             self._create_branch_fn(x, i, hardwts)
#             for i in range(self.n_ops)
#         ]
#         return tf.switch_case(index, branch_fns)