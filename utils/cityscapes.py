import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transform


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
    
def get_loaders(batch_size = 32, subclass = 'combined'):
    data_path = '/Users/nathanieljames/Desktop/dl_final_proj/target_fog'
    data_path = os.path.join(data_path, subclass)
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    transform_init = transform.Compose([transform.ToTensor()])
    dataset = CityscapesDataset(image_dir=train_path, cut_half=True, transform=transform_init)
    val_dataset = CityscapesDataset(image_dir=val_path, cut_half=True, transform=transform_init)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset)
    return dataloader, valloader