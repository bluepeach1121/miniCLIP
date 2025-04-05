import os
import ast
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class Flickr30kDataset(Dataset):
    """
    A PyTorch Dataset for the Flickr30k dataset, which expects:
      - A CSV file with columns: ['raw', 'sentids', 'split', 'filename', 'img_id']
      - A root directory containing the image files.
    """
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        split: str = "train",
        transform=None
    ):

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir

        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        captions_list = ast.literal_eval(row["raw"])

        if len(captions_list) == 5:
            text_caption = random.choice(captions_list)
        else:
            text_caption = captions_list[0] if len(captions_list) > 0 else ""

        if self.transform:
            image = self.transform(image)

        return image, text_caption
