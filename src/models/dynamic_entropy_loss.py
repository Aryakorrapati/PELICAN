import torch
import math

class DynamicEntropyLoss:
    def __init__(self, total_epochs, start_smoothing=0.11, final_smoothing=0):
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
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        raw_smoothing = self.final_smoothing + (self.start_smoothing - self.final_smoothing) * cosine_decay
        return max(0.01, raw_smoothing)  # Add floor to prevent total loss of regularization
