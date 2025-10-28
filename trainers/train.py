import importlib
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_model(config):
    module = importlib.import_module(config['path'])
    model_class = getattr(module, config['class_name'])
    return model_class(**config.get('params', {}))

def train_model(model_config, epochs=1):
    model = load_model(model_config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    dataset = datasets.MNIST('.', train=True, download=True,
                             transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{model_config['name']}] Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}")
    return model
