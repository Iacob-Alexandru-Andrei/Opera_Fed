import torch
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from bisect import bisect_right

def build_scheduler(scheduler, num_epochs, warmup_epochs, optimizer, num_batches, decay_rate, decay_epochs):

    num_steps = int(num_epochs * num_batches)
    warmup_steps = int(warmup_epochs * num_batches)
    decay_steps = int(decay_epochs * num_batches)
    multi_steps = [i * num_batches for i in [700, ]]
    
    lr_scheduler = None
    if scheduler == 'steplr':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t = decay_steps,
            decay_rate = decay_rate,
            warmup_lr_init = 5e-7,
            warmup_t = warmup_steps,
            t_in_epochs = False
        )
    elif scheduler == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones = multi_steps,
            gamma= 0.1,
            warmup_lr_init = 5e-7,
            warmup_t = warmup_steps,
            t_in_epochs=False,
        )
    else:
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial = num_steps,
            t_mul=1.,
            lr_min = 2.5e-7,
            warmup_lr_init = 2.5e-7,
            warmup_t = warmup_steps,
            cycle_limit = 1,
            t_in_epochs = False,
        )
        
    return lr_scheduler

from timm.scheduler.scheduler import Scheduler
from bisect import bisect_right

class MultiStepLRScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
        super().__init__(optimizer, param_group_field="lr")
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        
        assert self.warmup_t <= min(self.milestones)
    
    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None