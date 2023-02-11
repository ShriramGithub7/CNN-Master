

class AlbumentationImageDataset(Dataset):
  def __init__(self, image_list, train=True):
    self.image_list = image_list
    self.augmented = A.Compose(
                  {A.HorizontalFlip(),
                   A.Normalize(mean, std),
                   A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,rotate_limit=45),
                   A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,min_height=16, min_width=16,
                             fill_value=np.mean(mean), mask_fill_value=None),
                   A.ToGray()
                   })
    self.norm = A.Normalize(mean, std)
    self.train = train
  
  def __len__(self):
    return len(self.image_list)
  
  def __getitem__(Self, i):
    image, label = self.inage_list[i]
    
    if self.train:
      #apply augumentation only for training
      image=self.augmented(image=np.array(image))['image']
    self:
      iage=self.norm(image=np.array(inage))['image']
    image=np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    return torch.totensor(image, dtype=torch.float), label
                       
    
 
