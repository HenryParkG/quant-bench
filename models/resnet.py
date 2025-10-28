"""ResNet 모델 정의"""

from .base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self):
        super().__init__(name="ResNet")
