
import numpy as np
import albumentations as A
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Dataset():   
    def __init__(self):
    
        self.BATCH_SIZE=4
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck')
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True )
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True)

class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train=True):
        self.image_list = image_list
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)
        self.augmented = A.Compose({
            A.HorizontalFlip(),
            A.Normalize(self.mean, self.std),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                             fill_value=0.473363, mask_fill_value=None),
            A.ToGray()
        })
        self.norm = A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        self.train = train
  
    def __len__(self):
        return len(self.image_list)
  
    def __getitem__(self, i):
        image, label = self.image_list[i]
    
        if self.train:
            # apply augmentation only for training
            image = self.augmented(image=np.array(image))['image']
        else:
            image = self.norm(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
        return torch.tensor(image, dtype=torch.float), label   


class DataLoader():
    def __init__(self, trainset, testset):
        self.BATCH_SIZE=4
        self.train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=self.BATCH_SIZE,
                                                      shuffle=False, num_workers=1)
    

        self.test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=self.BATCH_SIZE,
                                              shuffle=False, num_workers=1)
    
