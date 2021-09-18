import tensorflow as tf
from hanser.train.learner import Learner, cast


from tfnas.models.ppnas.search import Network


class DARTSLearner(Learner):

    def __init__(self, model: Network, criterion, optimizer_arch, optimizer_model,
                 grad_clip_norm=0.0, **kwargs):
        self.grad_clip_norm = grad_clip_norm
        super().__init__(model, criterion, (optimizer_arch, optimizer_model), **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer_arch, optimizer_model = self.optimizers
        model_params, arch_params = [
            model.trainable_variables[s] for s in model.param_splits()]

        (input, target), (input_search, target_search) = batch
        with tf.GradientTape() as tape:
            input_search = cast(input_search, self.dtype)
            logits_search = model(input_search, training=True)
            logits_search = cast(logits_search, tf.float32)
            per_example_loss = self.criterion(target_search, logits_search)
            loss_search = self.reduce_loss(per_example_loss)
        grads = tape.gradient(loss_search, arch_params)
        self.apply_gradients(optimizer_arch, grads, arch_params)

        with tf.GradientTape() as tape:
            input = cast(input, self.dtype)
            logits = model(input, training=True)
            logits = cast(logits, tf.float32)
            per_example_loss = self.criterion(target, logits)
            loss = self.reduce_loss(per_example_loss)

        grads = tape.gradient(loss, model_params)
        self.apply_gradients(optimizer_model, grads, model_params, self.grad_clip_norm)
        self.update_metrics(self.train_metrics, target, logits, per_example_loss)

    def eval_batch(self, batch):
        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = self.model(inputs, training=False)
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)
