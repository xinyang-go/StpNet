import cv2
import os
import torch
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, dir):
        super(ImageNetDataset, self).__init__()
        with open(f"{dir}/.../names.txt") as fp:
            lines = [l.strip().split(" ") for l in fp.readlines()]
            self.name_id = {}
            for n, idx in lines:
                self.name_id[n] = int(idx)
        self.fns = os.listdir(dir)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = torch.tensor([self.name_id[fn.split("/")[-1].split(" ")[0]]]).long()
        img = torch.from_numpy(cv2.imread(fn)).permute(2, 0, 1) / 255.

        return img, label

    def __len__(self):
        return len(self.fns)
