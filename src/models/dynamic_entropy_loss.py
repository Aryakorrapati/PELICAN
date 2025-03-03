import torch

class DynamicEntropyLoss:
    def __init__(self, total_epochs, start_smoothing=0.1, final_smoothing=0):
        self.total_epochs = total_epochs
        self.start_smoothing = start_smoothing
        self.final_smoothing = final_smoothing
        self.epoch = 0

    def update_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, predict, targets):
        smoothing = self.get_dynamic_smoothing(self.epoch)
        return torch.nn.CrossEntropyLoss(label_smoothing=smoothing)(predict, targets.long())

    def get_dynamic_smoothing(self, epoch):
        progress = epoch / self.total_epochs
        return (1 - progress) * self.start_smoothing + progress * self.final_smoothing
