import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, model, num_classes=None):
        super().__init__()
        self.model = model
        if num_classes:
            self._adjust_fc(num_classes)

    def _adjust_fc(self, num_classes):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
