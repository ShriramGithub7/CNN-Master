from torch.autograd import backward
'''Train CIFAR10 with PyTorch.'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm

#from models import *

parser = argparse.ArgumentParser(description="Pytorch CIFAR10 Training")
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
#args=parser.parse_args()

device= 'cuda' if torch.cuda.is_available() else 'cpu'
print("device -1 ", device)

best_acc = 0
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, l1, scheduler):
  model.train()
  pbar=tqdm(train_loader)
  correct=0
  processed=0
  num_loops=0
  train_loss=0
  
  for batch_idx, (data, target) in enumerate(pbar):
    print("device -2 ", device)
    #data, target = data.to(device), target.to(device)
    #Init
    optimizer.zero_grad()
    
    #Predict
    y_pred=model(data)
    
    #Calculate_loss
    loss = F.nll_loss(y_pred, target)
    l1 = 0
    lambda_l1 = 0.01
    
    if l1:
      for p in model.parameters():
        l1 = l1+p.abs().sum()
        
    loss = loss + lambda_l1 * l1
    
    #Backpropogation
    loss.backward()
    optimizer.step()
    
    train_loss += loss.item()
    
    #Update LR
    scheduler.step()
    
    #Update pbar-tqdm
    
    pred=y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed = len(data)
    
    num_loops += 1
    pbar.set_description(desc=f'Batch_id={batch_idx} Loss={train_loss/num_loops:.5f} Accuracy={100*correct/processed:0.2f}')

  return 100*correct/processed, train_loss/num_loops

def test(model, device, test_loader, print_misclassified):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified = []
    misclassified_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Identify misclassified images
            incorrect = pred.ne(target.view_as(pred))
            misclassified_images = misclassified.extend(data[incorrect])

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
    return 100. * correct / len(test_loader.dataset), test_loss, misclassified_images
  
def fit_model(net, train_data, test_data, num_epochs=20, l1=False, l2=False):
    training_acc, training_loss, testing_acc, testing_loss, misclassified_img = list(), list(), list(), list(), list()

    if l2:
      optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    else:
      optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.017, epochs=num_epochs, steps_per_epoch=len(train_data))

    for epoch in range(1, num_epochs+1):
      print("EPOCH:", epoch)
      train_acc, train_loss = train(net, device, train_data, optimizer, l1, scheduler)
      test_acc, test_loss, misclassified_img = test(net, device, test_data)

      training_acc.append(train_acc)
      training_loss.append(train_loss)
      test_acc.append(test_acc)
      testing_loss.append(test_loss)
      misclassified_img.extend(misclassified_img)

    return net, (training_acc, training_loss, training_acc, testing_loss)

def print_misclassified(misclassified_img):
      misclassified_img = misclassified_img
      import matplotlib.pyplot as plt

      # Determine the number of rows and columns for the subplots
      rows = 2
      cols = 5

      # Use plt.subplots to create the subplots
      fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
      axs = axs.ravel()

      # Plot the misclassified images in the subplots
      for i, image in enumerate(misclassified_img[:10]):
          axs[i].imshow(image.squeeze().cpu().numpy(), cmap='gray_r')
          axs[i].set_title(f"True label: {target[i].item()}", fontweight='bold')

      # Remove unused subplots
      for i in range(10, rows * cols):
          fig.delaxes(axs[i])
      
      plt.suptitle("Misclassified Images", fontsize=15, fontweight='bold')

      # Show the plot
      plt.show()
               
                                        
