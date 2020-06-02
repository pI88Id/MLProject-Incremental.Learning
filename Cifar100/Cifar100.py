#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from tqdm import tqdm

class Cifar100:
    def __init__(self, batch_size=256, num_epochs=50, device='cuda', lr=1e-3, step_size=20, gamma=0.1):
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.DEVICE = device
        self.STEP_SIZE = step_size
        self.GAMMA = gamma
        self.LR = lr
        self.train_transform = transforms.Compose([
                                          transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes tensor with mean and standard deviation
                                         ])
        self.eval_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        self.DATA_DIR = 'Project-Cifar100'

        self.train_dataset = torchvision.datasets.CIFAR10(self.DATA_DIR, train=True, transform=self.train_transform, download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(self.DATA_DIR, train=False, transform=self.eval_transform, download=True)

        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

        # Check dataset sizes
        print('Cifar100 - DATASET CREATED')
        print('Train Dataset: {}'.format(len(self.train_dataset)))
        print('Test Dataset: {}'.format(len(self.test_dataset)))

    def load(self, split='train', index=0):

        data = []
        i = 0

        if split == 'train':

            searched_classes = np.linspace(index * 10, index * 10 + 9, 10)

            for el in self.train_dataset.targets:
                if el in searched_classes:
                    data.append(self.train_dataset[i])
                i += 1

            return DataLoader(data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        elif split == 'test':

            searched_classes = np.linspace(0, index * 10 + 9, (index + 1) * 10)

            for el in self.test_dataset.targets:
                if el in searched_classes:
                    data.append(self.train_dataset[i])
                i += 1

            return DataLoader(data, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

    def test(self, net, test_dataloader, criterion):

        net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        outputs = []
        labels = []
        for images, labels in tqdm(test_dataloader):
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)

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

    def plot(self, acc_train, acc_test, loss_train, loss_test):
        title = 'LossFunction - BATCH_SIZE= %d LR= %f  EPOCHS= %d  STEP_SIZE= %d GAMMA= %f' \
                % (self.BATCH_SIZE, self.LR, self.NUM_EPOCHS, self.STEP_SIZE, self.GAMMA)
        title2 = 'Accuracy classes - BATCH_SIZE= %d LR= %f  EPOCHS= %d  STEP_SIZE= %d GAMMA= %f' \
                 % (self.BATCH_SIZE, self.LR, self.NUM_EPOCHS, self.STEP_SIZE, self.GAMMA)

        x = np.linspace(1, self.NUM_EPOCHS, self.NUM_EPOCHS)

        plt.plot(x, loss_train, color='mediumseagreen')
        plt.plot(x, loss_test, color='lightseagreen')
        plt.title(title)
        plt.xticks(np.arange(1, self.NUM_EPOCHS, 4))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss_train', 'loss_test'], loc='best')
        plt.show()

        plt.plot(acc_train, color='mediumseagreen')
        plt.plot(acc_test, color='lightseagreen')
        plt.legend(['accuracy_train', 'accuracy_test'], loc='best')
        plt.title(title2)
        plt.xlabel('epoch')
        plt.ylabel('accuracy_score')

        print('Accuracy test', acc_test)
