import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms as T


def get_img_path(img_name, voc12_root):
    # return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.tif')
    return os.path.join(voc12_root, img_name + '.jpg')

# voc数据格式的dataset
class VOCImageDataset(Dataset):
    def __init__(self, img_name_list_path, voc_root, resize=None, num_class=5):
        self.voc_root = voc_root
        self.resize = resize
        self.num_class = num_class
        self.img_name_list, self.label_list = self.read_labeled_image_list(img_name_list_path, self.num_class)
        self.transform = transforms.Compose([
                transforms.Resize((self.resize, self.resize), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def read_labeled_image_list(self, data_list,num_classes):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] #+ '.tif'
            labels = np.zeros([num_classes], dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.
            img_name_list.append(image) #os.path.join(data_dir, image)

            img_labels.append(labels)
        return img_name_list, img_labels  # , np.array(img_labels, dtype=np.float32)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        label = self.label_list[idx]

        img = Image.open(get_img_path(name, self.voc_root)).convert('RGB')
        # img = T.ToTensor()(img)
        trans_img = self.transform(img)



        return {'name': name, 'img': trans_img, 'label': label}


class STEEL_ClsDataset(VOCImageDataset):
    def __init__(self, img_name_list_path, voc_root, resize=None, num_class=5):
        super().__init__(img_name_list_path,voc_root,resize=resize, num_class=num_class)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        return out