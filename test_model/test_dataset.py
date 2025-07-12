import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    """Dataset для тестирования на полных изображениях"""
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 2, normalize: str = '0-1'):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.normalize = normalize
        
        # Получение списка файлов
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(f"Mismatch in LR/HR file counts: {len(self.lr_files)} vs {len(self.hr_files)}")
        
        print(f"Найдено {len(self.lr_files)} пар изображений для тестирования")
    
    def __len__(self):
        return len(self.lr_files)
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Нормализует изображение в заданный диапазон"""
        if self.normalize == '0-1':
            return img.astype(np.float32) / 255.0
        elif self.normalize == '-1-1':
            return (img.astype(np.float32) / 127.5) - 1.0
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        # Загрузка изображений
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        
        if lr_img is None:
            raise IOError(f"Failed to read LR image: {lr_path}")
        if hr_img is None:
            raise IOError(f"Failed to read HR image: {hr_path}")
        
        # Конвертация BGR -> RGB
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Нормализация
        lr_img = self._normalize(lr_img)
        hr_img = self._normalize(hr_img)
        
        # Конвертация в тензор (HWC -> CHW)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).contiguous()
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).contiguous()
        
        return lr_tensor, hr_tensor, self.lr_files[idx]