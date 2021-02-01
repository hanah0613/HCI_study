##########################
### Import
##########################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
import PIL as Image

import numpy as np
import pandas as pd

from skimage import io
from tqdm import tqdm

##########################
### SETTINGS
##########################
# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

#PATH
PATH = './data/'
IMAGE_PATH = PATH+'images/'
TRAIN_PATH = IMAGE_PATH+'train/'
TEST_PATH = IMAGE_PATH+'test/'

# Others
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)

##########################
### PREPROCESS
##########################
#MAKE LABELS
with open(PATH+'label_v2.json') as file:
    data = json.load(file)
dataframe = pd.DataFrame(data['train'], columns=['filename', 'labels'])

classes = pd.unique(dataframe.labels)
for idx, row in dataframe.iterrows():
    dataframe['labels'][idx] = np.where(classes == row['labels'])[0][0]

##########################
### MAKE DATASET
##########################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

##########################
### Dataset
##########################


class FarmDataset(Dataset):
    def __init__(self, data, images_dir, transform=None):
        if type(data) is pd.DataFrame:
            self.dataframe = data
        else:
            self.dataframe = pd.DataFrame(data, columns=['filename', 'labels'])
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.images_dir,
            self.dataframe.iloc[idx, 0]
        )
        image = io.imread(img_name)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

##########################
### Model
##########################


class classifier(nn.Module):
    def __init__(self, D_out):
        super().__init__()
        self.conv = nn.Sequential(
            #3 224 128
            nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        #512 7 4

        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        self.classifier = nn.Linear(512, D_out)

    def forward(self, x):
        #print(x.size())
        features = self.conv(x)
        #print(features.size())
        x = self.avg_pool(features)
        #print(avg_pool.size())
        x = x.view(features.size(0), -1)
        #print(flatten.size())
        x = self.classifier(x)
        #x = self.softmax(x)
        return x


def main():
    #FOR TRAIN
    train_dataset = FarmDataset(
        data=dataframe, images_dir=TRAIN_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=, shuffle=True)

    model = classifier(14)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 15

    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            outputs.to(DEVICE)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = running_corrects.float() / len(train_loader)

        print('\n')
        print('epoch :', (e+1))
        print('training loss: {:.4f}, acc {:.4f} '.format(
            epoch_loss, epoch_acc.item()))


if __name__ == "__main__":
    main()
