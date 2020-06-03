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
        for _, images, labels in tqdm(test_dataloader):
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

    def plot(self, new_acc_train, new_acc_test, new_loss_train, new_loss_test, args):
        x = np.linspace(1, len(new_acc_train[0]), len(new_acc_train[0]))

        for i, (acc_train, acc_test, loss_train, loss_test) in enumerate(zip(new_acc_train, new_acc_test, new_loss_train, new_loss_test)):
            title = 'Accuracy dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  STEP_SIZE= %d GAMMA= %f' \
                    % (i+1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['STEP_SIZE'], args['GAMMA'])
            title2 = 'Loss dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  STEP_SIZE= %d GAMMA= %f' \
                     % (i+1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['STEP_SIZE'], args['GAMMA'])

            plt.plot(x, acc_train, color='mediumseagreen')
            plt.plot(x, acc_test, color='lightseagreen')
            plt.title(title)
            plt.xticks(np.arange(1, len(new_acc_train[0]), 4))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train accuracy', 'Test accuracy'], loc='best')
            plt.show()

            plt.plot(x, loss_train, color='mediumseagreen')
            plt.plot(x, loss_test, color='lightseagreen')
            plt.title(title2)
            plt.xticks(np.arange(1, len(new_acc_train[0]), 4))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train loss', 'Test loss'], loc='best')
            plt.show()

        print('Accuracy last test', new_acc_test[-1])
