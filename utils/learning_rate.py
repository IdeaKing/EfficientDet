import numpy as np


def updated_learning_rate(
        global_step,
        learning_rate_base,
        total_steps,
        num_warmup_steps=0,
        num_wait_steps=0):
    """Get learning rate."""
    if global_step < num_wait_steps:
        return 1e-9
    global_step = global_step - num_wait_steps
    if num_warmup_steps > total_steps:
        num_warmup_steps = total_steps - 1
    rate = cosine_decay_with_warmup(
        global_step,
        learning_rate_base,
        total_steps - num_wait_steps,
        warmup_steps=num_warmup_steps)
    return rate


def cosine_decay_with_warmup(
        global_step,
        learning_rate_base,
        total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError(
            "total_steps must be larger or equal to " "warmup_steps.")
    learning_rate = (
        0.5
        * learning_rate_base
        * (
            1
            + np.cos(
                np.pi
                * float(global_step - warmup_steps - hold_base_rate_steps)
                / float(total_steps - warmup_steps - hold_base_rate_steps)
            )
        )
    )
    if hold_base_rate_steps > 0:
        learning_rate = np.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate,
            learning_rate_base,
        )
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError(
                "learning_rate_base must be larger or equal to " "warmup_learning_rate."
            )
        slope = (learning_rate_base - warmup_learning_rate) / \
            float(warmup_steps)
        warmup_rate = slope * float(global_step) + warmup_learning_rate
        learning_rate = np.where(
            global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)
