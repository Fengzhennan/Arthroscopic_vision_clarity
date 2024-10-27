import os
import glob
import torch
import random
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms

class HemorrhageDataset(Dataset):

    def __init__(self, root="./dataset", transform=None):
        self.transform = transforms.Compose(transform)

        # from path get unpaired hemorrhage and clear frames
        self.hemorrhage_frames = glob.glob(os.path.join(root, "hemorrhage_frame/*"))
        self.clear_frames = glob.glob(os.path.join(root, "clear_frame/*"))

        self.use_cuda = torch.cuda.is_available()

        print(
            f"Number of hemorrhage frames: {len(self.hemorrhage_frames)}\n"
            f"Number of clean frames: {len(self.clear_frames)}\n"
        )

    def __getitem__(self, index):
        hemorrhage_image_path = random.choice(self.hemorrhage_frames)
        clear_image_path = self.clear_frames[index % len(self.clear_frames)]
        hemorrhage_image = self.transform(Image.open(hemorrhage_image_path))
        clear_image = self.transform(Image.open(clear_image_path))
        # hemorrhage_label = torch.zeros([1], dtype=torch.float,
        #                                requires_grad=False)
        # clear_label = torch.ones([1], dtype=torch.float,
        #                          requires_grad=False)

        return {"A": hemorrhage_image, "B": clear_image}

    def __len__(self):
        return max(len(self.clear_frames), len(self.hemorrhage_frames))


if __name__ == '__main__':
    from  torch.utils.data import DataLoader

    # calculate dataset's mean and sd

    root = r"../dataset"

    # half number of core numbers( i7-14700kf 20 cores, s.t. num_workers=10)
    dataloader = DataLoader(HemorrhageDataset(root, mode='train'),
                            batch_size=1,
                            shuffle=True,
                            )

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)