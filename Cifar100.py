import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

class Cifar100:
  def __init__(self, batch_size=256):
    
    self.train_transform = transforms.Compose([
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes tensor with mean and standard deviation
                                     ])
    self.eval_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                    
                                    ])
    self.DATA_DIR = 'Project-Cifar100'

    self.train_dataset = torchvision.datasets.CIFAR10(self.DATA_DIR, train=True, transform=self.train_transform, download=True)
    self.test_dataset = torchvision.datasets.CIFAR10(self.DATA_DIR, train=False, transform=self.eval_transform, download=True)

    self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Check dataset sizes
    print('Cifar100 - DATASET CREATED')
    print('Train Dataset: {}'.format(len(self.train_dataset)))
    print('Test Dataset: {}'.format(len(self.test_dataset)))

  def load(self, split='train'):
    
    if(split == 'train'):
      return self.train_dataloader
    elif(split == 'test'):
      return self.test_dataloader

