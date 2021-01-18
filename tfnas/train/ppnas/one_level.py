import tensorflow as tf
from hanser.train.learner import Learner, is_distribute_strategy, cast

from tfnas.models.ppnas.search import Network


def apply_gradients(learner, optimizer, grads, vars, grad_clip_norm=None):
    aggregate_grads_outside_optimizer = grad_clip_norm and is_distribute_strategy(learner._strategy)

    if aggregate_grads_outside_optimizer:
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)

    if grad_clip_norm:
        grads = tf.clip_by_global_norm(grads, grad_clip_norm)[0]
    if aggregate_grads_outside_optimizer:
        optimizer.apply_gradients(
            zip(grads, vars),
            experimental_aggregate_gradients=False)
    else:
        optimizer.apply_gradients(zip(grads, vars))


class PPNASLearner(Learner):

    def __init__(self, model: Network, criterion, optimizer_arch, optimizer_model,
                 add_arch_loss=False, **kwargs):
        self.add_arch_loss = add_arch_loss
        super().__init__(model, criterion, (optimizer_arch, optimizer_model), **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer_arch, optimizer_model = self.optimizers

        inputs, target = batch
        with tf.GradientTape() as tape:
            inputs = cast(inputs, self.dtype)
            logits = model(inputs, training=True)
            logits = cast(logits, tf.float32)
            per_example_loss = self.criterion(target, logits)
            loss = self.reduce_loss(per_example_loss)
            if self.add_arch_loss:
                arch_loss = model.arch_loss()
                arch_loss = tf.reduce_mean(arch_loss)
                loss = loss + arch_loss

        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        model_slice, arch_slice = model.param_splits()
        apply_gradients(self, optimizer_model, grads[model_slice], variables[model_slice], self.grad_clip_norm)
        apply_gradients(self, optimizer_arch, grads[arch_slice], variables[arch_slice])

        self.update_metrics(self.train_metrics, target, logits, per_example_loss)

    def eval_batch(self, batch):
        model = self.model

        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)

    def test_batch(self, inputs):
        model = self.model

        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, self.dtype)
        return preds
