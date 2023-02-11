import numpy as np
import albumentations as A
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class AlbumentationImageDataset(Dataset):
  def __init__(self, image_list, train=True):
    self.image_list = image_list
    self.augmented = A.Compose(
                  {A.HorizontalFlip(),
                   A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                   A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,rotate_limit=45),
                   A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,min_height=16, min_width=16,
                             fill_value=np.mean(0.473363), mask_fill_value=None),
                   A.ToGray()
                   })
    self.norm = A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    self.train = train
  
  def __len__(self):
    return len(self.image_list)
  
  def __getitem__(Self, i):
    image, label = self.inage_list[i]
    
    if self.train:
      #apply augumentation only for training
      image=self.augmented(image=np.array(image))['image']
    else:
      iage=self.norm(image=np.array(inage))['image']
    image=np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    return torch.totensor(image, dtype=torch.float), label
   
class data_loader():   
    

    BATCH_SIZE=4
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True )
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True)


    train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True), batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)
    

    test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False), batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=1)
    
    def view_data():
        def show(img):
          img = img/2 + 0.5
          npimg= img.numpy()
          plt.imshow(np.transpose(npimg, (1, 2, 0)))

        dataiter = iter(train_loader)
        images, labels= next(dataiter)


        show(torchvision.utils.make_grid(images, normalize=False))

        print(' '.join('%5s' % classes[labels[j]] for j in range (4)))

        images.shape
