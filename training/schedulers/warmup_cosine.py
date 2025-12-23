from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

class Scheduler(SequentialLR):
    def __init__(self, optimizer, warmup_iters=100, total_iters=1000, start_factor=0.001, eta_min=0.0, last_epoch=-1, verbose=False):
        schedulers = [
            LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters),
            CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=eta_min)
        ]
        super().__init__(optimizer, schedulers=schedulers, milestones=[warmup_iters], last_epoch=last_epoch, verbose=verbose)
