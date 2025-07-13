from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import random
from torchvision import transforms
class AlbumInferenceDataset(Dataset):
    def __init__(self, album_path, album_size=20):
        """
        Dataset để đọc một album ảnh để inference.

        Args:
            album_path: Đường dẫn thư mục chứa ảnh của album.
            album_size: Số lượng ảnh muốn lấy từ album.
            transform: Phép biến đổi áp dụng cho ảnh (resize, toTensor,...).
        """
        self.album_path = album_path
        self.album_size = album_size
        self.transform = transforms.Compose([
                            transforms.Resize((384, 384)),
                            transforms.ToTensor(),
                                            ])
        self.seed = 42

        self.image_paths = [f for f in os.listdir(album_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        random.seed(self.seed + idx)

        if not self.image_paths:
            images = [torch.zeros(3, 384, 384)] * self.album_size
        else:
            if len(self.image_paths) < self.album_size:
                selected_images = random.choices(self.image_paths, k=self.album_size)
            else:
                selected_images = random.sample(self.image_paths, k=self.album_size)
            
            images = []
            for image_path in selected_images:
                image = Image.open(os.path.join(self.album_path, image_path)).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)

        return torch.stack(images)
