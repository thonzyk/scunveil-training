import tensorflow as tf


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, finish_lr, decay_steps):
        super(WarmupSchedule, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.finish_lr = finish_lr
        self.decay_steps = decay_steps

    def update_lr(self, lr):
        self.base_lr = lr

    def __call__(self, step):
        base_lr = tf.cast(self.base_lr, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        finish_lr = tf.cast(self.finish_lr, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        step = tf.cast(step, tf.float32)

        def warmup_lr():
            return base_lr * (step / warmup_steps)

        def decay_lr():
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.141592653589793) * (step - warmup_steps) / decay_steps))
            return finish_lr + (base_lr - finish_lr) * cosine_decay

        def constant_lr():
            return finish_lr

        learning_rate = tf.cond(
            step < warmup_steps,
            warmup_lr,
            lambda: tf.cond(
                step < warmup_steps + decay_steps,
                decay_lr,
                constant_lr
            )
        )

        return learning_rate
    

class SamplesSeen(tf.keras.callbacks.Callback):
    def __init__(self, batch_size: int, key: str = "samples_seen"):
        self.bs = int(batch_size)
        self.key = key
        self.n = 0

    def on_train_batch_end(self, batch, logs=None):
        self.n += self.bs  # accumulate silently

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs[self.key] = self.n
