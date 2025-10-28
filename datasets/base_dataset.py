from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, batch_size=64, train=True):
        self.batch_size = batch_size
        self.train = train
        self.loader = self._load_dataset()

    @abstractmethod
    def _load_dataset(self):
        pass

    def get_loader(self):
        return self.loader
