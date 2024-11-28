import numpy as np
import torch.optim as optim
from torch.optim.optimizer import Optimizer

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup=50, max_iters=100):
        self.warmup_epoch = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup_epoch:
            lr_factor *= epoch * 1.0 / self.warmup_epoch
        return lr_factor
        
class ReduceLROnValidMetric(object):

    def __init__(self, optimizer, mode='min', factor=0.1, 
                 min_lr=0, start_value=3, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.mode = mode
        self.last_epoch = 0
        self.start_value = start_value
        self.eps = eps

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if current < self.start_value:
            pass 
        else:
            lr_factor = self.factor * (1 + np.cos(np.pi * (current - self.start_value)))
            self._reduce_lr(lr_factor)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, lr_factor):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * lr_factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    warmup = 100
    max_iters = 100
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup=warmup, max_iters=max_iters)

    # Plotting
    epochs = list(range(2000))
    # sns.set()
    
    x = plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title(f"Cosine Warm-up Learning Rate Scheduler warmup={warmup}, max_iter={max_iters}")
    print()
    # plt.savefig()
