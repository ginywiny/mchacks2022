from statistics import mode
from typing import Tuple
from unittest import loader
import torch
from torch import device
import torch.cuda
import torch.optim
import torch.nn
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18
from torchvision import datasets, transforms as T
from torchvision.datasets.folder import make_dataset
from torchvision.datasets import ImageFolder
import time
import copy
from PIL import Image

def load_model():

    model = resnet18(pretrained=True)
    return model


def train_model(model, dataloader):
    #define the loss fn and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #initialize empty list to track batch losses
    batch_losses = []

    model.train()

    #train the neural network for 5 epochs
    for epoch in range(5):

        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
                    
            #reset gradients
            optimizer.zero_grad()
            
            #forward propagation through the network
            out = model(img)
            
            #calculate the loss
            loss = criterion(out, label)
            
            #track batch loss
            batch_losses.append(loss.item())
            
            #backpropagation
            loss.backward()
            
            #update the parameters
            optimizer.step()
    
    model.eval()
    return model


def main():
    dataset_dir_name = "dataset"

    dataset = ImageFolder(
        dataset_dir_name,
        transform=T.Compose([
            T.ToTensor()
        ]),
        # target_transform=T.Compose([
        #     T.ToTensor()
        # ]),
        loader=Image.open
    )
    
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = load_model()
    num_in_features = model.fc.in_features

    # replace fully connected layer with one who's output dimension matches the number of classes
    model.fc = torch.nn.Linear(num_in_features, len(dataset.classes))

    train_model(model, dataloader)


if __name__ == "__main__":
    main()