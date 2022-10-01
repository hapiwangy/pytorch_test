# 好像怪怪der
# 使用pytorch-lightning來向tensorflow依樣可以使用少少的行數
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torchvision.datasets import MNIST
PATH_DATASETS = ""
AVAIL_GPUS = min(1, torch.cuda.device_count()) # 使用gpu or cpu
BATCH_SIZE = 256 if AVAIL_GPUS else 64 # 批量

# 建立模型
class MNISTmodel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(28* 28, 10)
    def forward(self, x):
        # relu activation + 完全連接層
        return torch.relu(self.l1(x.view(x.size(0), -1)))
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss
    def configure_optimizer(self):
        # 設定adam優化器
        return torch.optim.Adam(self.parameters(), lr=0.02)

# 訓練模型
## 先下載MNIST訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
## 建立模型物件
mnist_model = MNISTmodel()
## 建立DATALOADER:這是一種python generator，一次載入固定的訓練資料到memory裡面，可以節省記憶體使用
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
## 使用fit進行訓練
trainer = Trainer(accelerator = 'gpu', devices=AVAIL_GPUS, max_epochs=3)
trainer.fit(mnist_model, train_loader)
trainer.test()