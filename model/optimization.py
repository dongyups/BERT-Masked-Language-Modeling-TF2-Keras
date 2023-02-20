import tensorflow as tf
import tensorflow_addons as tfa
from typing import Mapping, Any, Union, Optional


### LR from Attention is all you need (no use) ###
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, dtype=tf.float32))
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


### LR from BERT pre-training ###
class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    reference: https://github.com/tensorflow/models/blob/v2.11.3/official/modeling/optimization/lr_schedule.py#L92-L162
    Linear warmup schedule.
    
    Add linear warmup schedule to a learning rate schedule.
    warmup_lr is the initial learning rate, the final learning rate of the
    init_warmup period is the initial learning rate of lr_schedule in use.
    The learning rate at each step linearly increased according to the following
    formula:
        learning_rate = warmup_lr + step / warmup_steps
                      * (final_warmup_lr - warmup_lr).
    Using warmup overrides the learning rate schedule by the number of warmup steps.
    Args:
        after_warmup_lr_sched: tf.keras.optimizers.schedules .LearningRateSchedule or a constant.
        warmup_steps: Number of the warmup steps.
        warmup_learning_rate: Initial learning rate for the warmup.
        name: Optional, name of warmup schedule.
    """
    def __init__(self,
        after_warmup_lr_sched: Union[
        tf.keras.optimizers.schedules.LearningRateSchedule, float],
        warmup_steps: int,
        warmup_learning_rate: float,
        name: Optional[str] = None):
        super().__init__()
        self._name = name
        self._after_warmup_lr_sched = after_warmup_lr_sched
        self._warmup_steps = warmup_steps
        self._init_warmup_lr = warmup_learning_rate
        if isinstance(after_warmup_lr_sched, tf.keras.optimizers.schedules.LearningRateSchedule):
            self._final_warmup_lr = after_warmup_lr_sched(warmup_steps)
        else:
            self._final_warmup_lr = tf.cast(after_warmup_lr_sched, dtype=tf.float32)

    def __call__(self, step: int):
        global_step = tf.cast(step, dtype=tf.float32)

        linear_warmup_lr = (
            self._init_warmup_lr + global_step / self._warmup_steps * 
            (self._final_warmup_lr - self._init_warmup_lr))

        if isinstance(self._after_warmup_lr_sched, tf.keras.optimizers.schedules.LearningRateSchedule):
            after_warmup_lr = self._after_warmup_lr_sched(step)
        else:
            after_warmup_lr = tf.cast(self._after_warmup_lr_sched, dtype=tf.float32)

        lr = tf.cond(global_step < self._warmup_steps,
                    lambda: linear_warmup_lr,
                    lambda: after_warmup_lr)
        return lr

    def get_config(self) -> Mapping[str, Any]:
        if isinstance(self._after_warmup_lr_sched, tf.keras.optimizers.schedules.LearningRateSchedule):
            config = {
                "after_warmup_lr_sched": self._after_warmup_lr_sched.get_config()
                }  # pytype: disable=attribute-error
        else:
            config = {
                "after_warmup_lr_sched": self._after_warmup_lr_sched
                }  # pytype: disable=attribute-error

        config.update({
            "warmup_steps": self._warmup_steps,
            "warmup_learning_rate": self._init_warmup_lr,
            "name": self._name
        })
        return config


### AdamW optimizer with warmup and linear decay LR ###
def CreateOptimizer(
    warmup_steps,
    num_train_steps,
    initial_learning_rate=1e-4,
    weight_decay_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    ):
    """
    Default settings for BERT pre-training:
        Adam with L2 weight decay
        weight_decay_rate = 0.01
        learning_rate = 1e-4
        beta_1 = 0.9
        beta_2 = 0.999
        warmup_steps = 10,000
        total_steps = 1,000,000
        epochs = 40
        mini-batch size = 256
    Each epoch has 25,000 steps and 0.01 of the total training steps is used for initializing the learning rate.
    """
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False
    )
    warmup_schedule = LinearWarmup(
        warmup_learning_rate = 0,
        after_warmup_lr_sched = linear_decay,
        warmup_steps = warmup_steps
    )
    return tfa.optimizers.AdamW(
        learning_rate=warmup_schedule,
        weight_decay=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=[
            "LayerNormalization", "layer_normalization", "LayerNorm", "layer_norm", 
            "OutputBias", "output_bias", "Bias", "bias",
            ])
