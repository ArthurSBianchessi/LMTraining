from torch.optim.lr_scheduler import LambdaLR

class Scheduler(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, lr_lambda=lambda _: 1.0, last_epoch=last_epoch)
