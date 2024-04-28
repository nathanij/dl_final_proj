import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset


class CityscapesDataset(Dataset):
    def __init__(self, image_dir, cut_half = True, transform = None):
        self.image_dir = image_dir
        self.imgs = os.listdir(image_dir)

        self.cut_half = cut_half
        self.transforms = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_mask = Image.open(os.path.join(self.image_dir, self.imgs[idx]))
        if self.cut_half:
            x_width, y_height = img_mask.size
            split = x_width / 2

            img = img_mask.crop((0, 0, split, y_height))

            mask = img_mask.crop((split, 0, split + split, y_height))

            if self.transforms:
                img = self.transforms(img)
                mask = self.transforms(mask)

            return img, mask

        return img_mask