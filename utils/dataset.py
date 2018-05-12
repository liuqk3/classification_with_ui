import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader,Dataset
class attribute_dataset(Dataset):

    def __init__(self,labels,root_dir,transform=None):
        self.labels=labels
        self.root_dir=root_dir
        self.transform=transform
        self.label_shift = {'skirt_length_labels': 0,
                            'coat_length_labels': 6,
                            'collar_design_labels': 14,
                            'lapel_design_labels': 19,
                            'neck_design_labels': 24,
                            'neckline_design_labels': 29,
                            'pant_length_labels': 39,
                            'sleeve_length_labels': 45
                            }
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.labels.iloc[idx,0]
        full_path = os.path.join(self.root_dir,image_name)
        image = Image.open(full_path)
        attribute=self.labels.iloc[idx,1]
        labels = self.labels.iloc[idx,2]
        if self.transform is not None:
            image=self.transform(image)
        label=labels.index('y')
        label=label+self.label_shift[attribute]
        return [image,label]
