import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from tqdm import tqdm

class Cifar100:
    def __init__(self, batch_size=256):

        self.train_transform = transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes tensor with mean and standard deviation
                                         ])
        self.eval_transform = transforms.Compose([
                                         transforms.Resize(256),
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

    def test(self, net, test_dataloader, DEVICE, criterion):

        net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        outputs = []
        labels = []
        for images, labels in tqdm(test_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward Pass
            outputs = net(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / float(len(test_dataloader))
        loss = criterion(outputs, labels)

        return accuracy, loss
