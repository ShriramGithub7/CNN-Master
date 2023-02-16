import numpy as np
import albumentations as A
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class CIFAR10Dataset(Dataset):
    BATCH_SIZE = 64
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck')
    
    def __init__(self, train=True):
        self.train = train
        
        if self.train:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
            
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)

        self.augmented = A.Compose([
            A.HorizontalFlip(),
            A.Normalize(self.mean, self.std),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,min_height=16, min_width=16,
                           fill_value=np.mean(self.mean), mask_fill_value=None),
            A.ToGray()
        ])

    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, i):
        image, label = self.dataset[i]
        
        if self.train:
            # apply augmentation only for training
            image = self.augmented(image=np.array(image))['image']
        else:
            norm = A.Normalize(self.mean, self.std)
            image = norm(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
        return torch.tensor(image, dtype=torch.float), label 
    
    def data_stats(self):
        images = np.concatenate([np.array(self[i][0]) for i in range(len(self))])
        print(' - Num Images:', len(self))
        print(' - min:', np.min(images, axis=(0,1,2)) / 255.)
        print(' - max:', np.max(images, axis=(0,1,2)) / 255.)
        print(' - mean:', np.mean(images, axis=(0,1,2)) / 255.)
        print(' - std:', np.std(images, axis=(0,1,2)) / 255.)
        print(' - var:', np.var(images, axis=(0,1,2)) / 255.) 


class CIFAR10DataLoader():
    def __init__(self, trainset, testset):
        self.BATCH_SIZE=64
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.BATCH_SIZE,
                                                      shuffle=False, num_workers=1)
    
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.BATCH_SIZE,
                                              shuffle=False, num_workers=1)
    
    def show_sample(self, trainloader):
        classes = CIFAR10Dataset.classes
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, ax in enumerate(axs.flatten()):
            image = np.transpose(images[i], (1, 2, 0))
           
