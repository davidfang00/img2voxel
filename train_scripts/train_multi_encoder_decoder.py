import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ShapeNetDataset import MultiShapeNetDataset
from models import MultiEncoderDecoder
from eval import calc_average_iou
from train_func import train_model

datasets_dir = 'data'

train_img_dataset_dir = datasets_dir + '/train_imgs'
val_img_dataset_dir = datasets_dir + '/val_imgs'
test_img_dataset_dir = datasets_dir + '/test_imgs'

train_voxel_dataset_dir = datasets_dir + '/train_voxels'
val_voxel_dataset_dir = datasets_dir + '/val_voxels'
test_voxel_dataset_dir = datasets_dir + '/test_voxels'

print(datasets_dir)

# For reproducibility
torch.manual_seed(1234)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

print(len(list(os.listdir(train_img_dataset_dir))))
print(len(list(os.listdir(train_voxel_dataset_dir))))

print(len(list(os.listdir(val_img_dataset_dir))))
print(len(list(os.listdir(val_voxel_dataset_dir))))

print(len(list(os.listdir(test_img_dataset_dir))))
print(len(list(os.listdir(test_voxel_dataset_dir))))

import scipy.io
from PIL import Image
import random

num_images_per_model = 12

train_dataloader = DataLoader(MultiShapeNetDataset('train', 12, downsample_size = 32), batch_size=4, shuffle=True, num_workers=2)
test_dataloader = DataLoader(MultiShapeNetDataset('test', 12, downsample_size = 32), batch_size=4, shuffle=True, num_workers=2)
val_dataloader = DataLoader(MultiShapeNetDataset('val', 12, downsample_size = 32), batch_size=4, shuffle=True, num_workers=2)

print(len(train_dataloader), len(test_dataloader), len(val_dataloader))

embedding_dims = 64

model_1 = MultiEncoderDecoder(embedding_dims).to(device) 
model_1.load_state_dict(torch.load('models/multi_encoder_decoder_weights_best_val_acc64.pt'))

dataloaders = {'test': test_dataloader, 'train': train_dataloader, 'val': val_dataloader}
criterion = nn.BCELoss()

learning_rate = 8e-5
optimizer_1 = optim.Adam(model_1.parameters(), lr=learning_rate)

save_dir = 'models/'
save_all_epochs = False
num_epochs = 3

trained_model_1, validation_history_1, train_history_1 = train_model(model=model_1, 
                                                                     dataloaders=dataloaders, 
                                                                     criterion=criterion, 
                                                                     optimizer=optimizer_1,
                                                                     save_dir=save_dir, 
                                                                     save_all_epochs=save_all_epochs, 
                                                                     num_epochs=num_epochs)

print('Train History:', train_history_1)
print('Val History:', validation_history_1)

eval_model = trained_model_1
eval_model.eval()

print('Test IoU:', calc_average_iou(eval_model, test_dataloader))
