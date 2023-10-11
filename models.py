import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.encoder = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, embedding_dims)

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class VoxelEncoder(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, 5)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(64, 128, 5)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv3d(128, 64, 5, stride = 2)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv3d(64, 32, 5, stride = 3)
        self.relu4 = nn.PReLU()

        self.fc = nn.Linear(256, embedding_dims)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
    
class VoxelDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, 32, 7)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.ConvTranspose3d(32, 128, 7)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.ConvTranspose3d(128, 256, 7)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.ConvTranspose3d(256, 128, 7)
        self.relu4 = nn.PReLU()
        self.conv5 = nn.ConvTranspose3d(128, 1, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], 1, 4, 4, 4)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x
    
class EncoderDecoder(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        # self.encoder = ImageEncoder(embedding_dims)

        self.encoder = VoxelEncoder(embedding_dims)
        self.decoder = VoxelDecoder(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MultiEncoderDecoder(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.encoder = ImageEncoder(embedding_dims)

        # self.encoder = VoxelEncoder(embedding_dims)
        self.decoder = VoxelDecoder(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        res = torch.zeros((B, 1, 32, 32, 32))
        
        for i in range(B):
            multi_im = x[i]
            multi_im = self.encoder(multi_im)            
            multi_im = self.decoder(multi_im)
            multi_im = torch.sum(multi_im, axis = 0)
            res[i] = multi_im
        
        res = self.sigmoid(res)
        return res

class ImageEncoderDecoder(nn.Module):
    def __init__(self, embedding_dims, image_encoder_path, voxel_encoder_path, finetuned = False):
        super().__init__()
        
        if finetuned:
            image_encoder_model = ImageEncoder(embedding_dims)
            image_encoder_model.load_state_dict(torch.load(image_encoder_path))
            self.encoder = image_encoder_model

            voxel_encoder_model = EncoderDecoder(embedding_dims)
            voxel_encoder_model.load_state_dict(torch.load(voxel_encoder_path))
            self.decoder = voxel_encoder_model.decoder
        else:
            self.encoder = ImageEncoder(embedding_dims)
            self.decoder = VoxelDecoder(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x