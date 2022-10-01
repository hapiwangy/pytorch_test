import torch
import os
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = MNIST("", train=True, download=True,transform=transforms.ToTensor())