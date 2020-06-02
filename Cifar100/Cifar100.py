import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np

class Cifar100:
  def __init__(self, batch_size=256):

    self.batch_size = batch_size
    
    self.train_transform = transforms.Compose([
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes tensor with mean and standard deviation
                                     ])
    self.eval_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                    
                                    ])
    self.DATA_DIR = 'Project-Cifar100'

    self.train_dataset = torchvision.datasets.CIFAR100(self.DATA_DIR, train=True, transform=self.train_transform, download=True)
    self.test_dataset = torchvision.datasets.CIFAR100(self.DATA_DIR, train=False, transform=self.eval_transform, download=True)

    # Check dataset sizes
    print('Cifar100 - DATASET CREATED')
    print('Train Dataset: {}'.format(len(self.train_dataset)))
    print('Test Dataset: {}'.format(len(self.test_dataset)))

  def load(self, split='train', index=0):

    data = []
    i = 0
    
    if(split == 'train'):
      
      searched_classes = np.linspace(index*10, index*10 + 9, 10)

      for el in self.train_dataset.targets:
        if (el in searched_classes):
          data.append(self.train_dataset[i])
        i+=1

      return DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    elif(split == 'test'):
      
      searched_classes = np.linspace(0, index*10 + 9, (index + 1)*10)

      for el in self.test_dataset.targets:
        if (el in searched_classes):
          data.append(self.train_dataset[i])
        i+=1

      return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=4)

