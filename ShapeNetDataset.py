from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import scipy.io

datasets_dir = 'data'

train_img_dataset_dir = datasets_dir + '/train_imgs'
val_img_dataset_dir = datasets_dir + '/val_imgs'
test_img_dataset_dir = datasets_dir + '/test_imgs'

train_voxel_dataset_dir = datasets_dir + '/train_voxels'
val_voxel_dataset_dir = datasets_dir + '/val_voxels'
test_voxel_dataset_dir = datasets_dir + '/test_voxels'

class ShapeNetDataset(Dataset):
    def __init__(self, stage, num_images_per_model, downsample_size = 32):

        if stage == 'train':
            self.image_dir = train_img_dataset_dir
            self.voxel_dir = train_voxel_dataset_dir
            self.num_models_to_use = 33000
        elif stage == 'val':
            self.image_dir = val_img_dataset_dir
            self.voxel_dir = val_voxel_dataset_dir
            self.num_models_to_use = 4900
        elif stage == 'test':
            self.image_dir = test_img_dataset_dir
            self.voxel_dir = test_voxel_dataset_dir
            self.num_models_to_use = 10010
        else:
            assert(False)

        self.num_images_per_model = num_images_per_model
        self.img_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias = True),
            transforms.CenterCrop(size=256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.image_model_names = set(os.listdir(self.image_dir))
        self.voxel_model_names = set(os.listdir(self.voxel_dir))
        self.model_names = sorted(self.image_model_names.intersection(self.voxel_model_names))[:self.num_models_to_use]
        self.image_names = sorted(os.listdir(os.path.join(self.image_dir, self.model_names[0])))

        self.num_images = len(self.model_names) * self.num_images_per_model

        self.downsample_size = downsample_size

        print(len(self.image_model_names), len(self.voxel_model_names), len(self.model_names), 
              min(self.model_names) + ' to ' + max(self.model_names))
        
    def __len__(self):
        return self.num_models_to_use * self.num_images_per_model
        return self.num_images

    def __getitem__(self, idx):
        model_num = idx // self.num_images_per_model
        img_num = idx % self.num_images_per_model
        
        img_path = os.path.join(self.image_dir, self.model_names[model_num]) + '/' + self.image_names[img_num]
        voxel_path = os.path.join(self.voxel_dir, self.model_names[model_num]) + '/model.mat'

        voxel_dict = scipy.io.loadmat(voxel_path)
        voxel = voxel_dict['input']
        voxel = torch.tensor(voxel, dtype = torch.float)

        if self.downsample_size:
            voxel = torch.nn.functional.interpolate(voxel.unsqueeze(0), 
                                                    (self.downsample_size, self.downsample_size, self.downsample_size), 
                                                    mode = 'trilinear')
            voxel = voxel.squeeze(0)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img.load()

        img = transforms.ToTensor()(img)[:3, ...]
        if self.img_transforms:
            img = self.img_transforms(img)

        return img.float(), voxel.float(), model_num, img_num
    
class MultiShapeNetDataset(Dataset):
    def __init__(self, stage, num_images_per_model, downsample_size = 32):

        if stage == 'train':
            self.image_dir = train_img_dataset_dir
            self.voxel_dir = train_voxel_dataset_dir
            self.num_models_to_use = 210
        elif stage == 'val':
            self.image_dir = val_img_dataset_dir
            self.voxel_dir = val_voxel_dataset_dir
            self.num_models_to_use = 30
        elif stage == 'test':
            self.image_dir = test_img_dataset_dir
            self.voxel_dir = test_voxel_dataset_dir
            self.num_models_to_use = 60
        else:
            assert(False)

        self.num_images_per_model = num_images_per_model
        self.img_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias = True),
            transforms.CenterCrop(size=256),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.image_model_names = set(os.listdir(self.image_dir))
        self.voxel_model_names = set(os.listdir(self.voxel_dir))
        self.model_names = sorted(self.image_model_names.intersection(self.voxel_model_names))[:self.num_models_to_use]
        self.image_names = sorted(os.listdir(os.path.join(self.image_dir, self.model_names[0])))

        self.num_images = len(self.model_names) * self.num_images_per_model

        self.downsample_size = downsample_size

        print(len(self.image_model_names), len(self.voxel_model_names), len(self.model_names), 
              min(self.model_names) + ' to ' + max(self.model_names))
        
    def __len__(self):
        return self.num_models_to_use
        # return self.num_models_to_use * self.num_images_per_model
        # return self.num_images

    def __getitem__(self, idx):
#         model_num = idx // self.num_images_per_model
#         img_num = idx % self.num_images_per_model
        
#         img_path = os.path.join(self.image_dir, self.model_names[model_num]) + '/' + self.image_names[img_num]

        model_num = idx
        voxel_path = os.path.join(self.voxel_dir, self.model_names[model_num]) + '/model.mat'

        voxel_dict = scipy.io.loadmat(voxel_path)
        voxel = voxel_dict['input']
        voxel = torch.tensor(voxel, dtype = torch.float)

        if self.downsample_size:
            voxel = torch.nn.functional.interpolate(voxel.unsqueeze(0), 
                                                    (self.downsample_size, self.downsample_size, self.downsample_size), 
                                                    mode = 'trilinear')
            voxel = voxel.squeeze(0)

        images = []
        for i in range(len(self.image_names)):
            img_path = os.path.join(self.image_dir, self.model_names[model_num]) + '/' + self.image_names[i]
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img.load()

            img = transforms.ToTensor()(img)[:3, ...]
            if self.img_transforms:
                img = self.img_transforms(img)
            images.append(img)
            
        images = torch.stack(images)

        return images.float(), voxel.float(), self.model_names[model_num]