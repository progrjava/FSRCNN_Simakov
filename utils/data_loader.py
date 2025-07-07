import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from typing import Tuple, List, Optional

class PatchDataset(Dataset):
    """
    Dataset для обработки изображений: разбиение на патчи, нормализация и аугментация.
    
    Args:
        lr_dir (str): Путь к папке с изображениями низкого разрешения
        hr_dir (str): Путь к папке с изображениями высокого разрешения
        patch_size (int): Размер патча для LR изображений (HR патчи будут 2*patch_size)
        scale (int): Коэффициент увеличения разрешения (по умолчанию 2)
        augment (bool): Применять ли аугментацию данных (по умолчанию True)
        normalize (str): Тип нормализации: '0-1' или '-1-1' (по умолчанию '0-1')
    """
    def __init__(self, 
                 lr_dir: str, 
                 hr_dir: str, 
                 patch_size: int = 48, 
                 scale: int = 2,
                 augment: bool = True,
                 normalize: str = '0-1'):
        
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale
        self.scale = scale
        self.augment = augment
        self.normalize = normalize
        
        # Проверка существования директорий
        if not os.path.exists(lr_dir):
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")
        
        # Получение списка файлов
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        
        # Проверка соответствия файлов
        if len(self.lr_files) != len(self.hr_files):
            raise ValueError("Mismatch in LR/HR file counts")
        if self.lr_files != self.hr_files:
            raise ValueError("Mismatch in LR/HR filenames")
        
        # Предварительная загрузка метаданных изображений
        self.image_pairs = []
        for lr_name, hr_name in zip(self.lr_files, self.hr_files):
            lr_path = os.path.join(lr_dir, lr_name)
            hr_path = os.path.join(hr_dir, hr_name)
            
            lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            
            if lr_img is None:
                raise IOError(f"Failed to read LR image: {lr_path}")
            if hr_img is None:
                raise IOError(f"Failed to read HR image: {hr_path}")
                
            # Проверка соотношения размеров
            h, w = lr_img.shape[:2]
            h_hr, w_hr = hr_img.shape[:2]
            
            if h_hr != h * scale or w_hr != w * scale:
                raise ValueError(
                    f"Size mismatch: LR={w}x{h}, HR={w_hr}x{h_hr}, scale={scale}"
                )
            
            self.image_pairs.append((lr_path, hr_path, w, h))
        
        # Расчет общего количества патчей
        self.num_patches = 0
        self.patch_coords = []
        
        for lr_path, hr_path, w, h in self.image_pairs:
            # Количество патчей по ширине и высоте
            w_patches = w // patch_size
            h_patches = h // patch_size
            
            for i in range(h_patches):
                for j in range(w_patches):
                    self.patch_coords.append((
                        lr_path,
                        hr_path,
                        j * patch_size,  # x_start
                        i * patch_size,   # y_start
                    ))
            
            self.num_patches += w_patches * h_patches

    def __len__(self) -> int:
        """Возвращает общее количество патчей."""
        return self.num_patches

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Нормализует изображение в заданный диапазон."""
        if self.normalize == '0-1':
            return img.astype(np.float32) / 255.0
        elif self.normalize == '-1-1':
            return (img.astype(np.float32) / 127.5) - 1.0
        else:
            raise ValueError(f"Unknown normalization: {self.normalize}")

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Применяет случайные аугментации к паре патчей."""
        # Вертикальное отражение
        if random.random() > 0.5:
            lr = np.flipud(lr)
            hr = np.flipud(hr)
        
        # Горизонтальное отражение
        if random.random() > 0.5:
            lr = np.fliplr(lr)
            hr = np.fliplr(hr)
        
        # Повороты на 90, 180, 270 градусов
        rotation = random.choice([0, 1, 2, 3])
        if rotation != 0:
            lr = np.rot90(lr, rotation)
            hr = np.rot90(hr, rotation)
        
        return lr, hr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает пару патчей (LR, HR) по индексу.
        
        Args:
            idx (int): Индекс патча в датасете
            
        Returns:
            tuple: (LR патч, HR патч) в формате torch.Tensor
        """
        lr_path, hr_path, x, y = self.patch_coords[idx]
        
        # Загрузка изображений
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        
        # Извлечение патчей
        lr_patch = lr_img[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]
        
        hr_patch = hr_img[
            y * self.scale:(y + self.patch_size) * self.scale,
            x * self.scale:(x + self.patch_size) * self.scale
        ]
        
        # Конвертация в RGB (OpenCV загружает в BGR)
        lr_patch = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2RGB)
        hr_patch = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
        
        # Аугментация
        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)
        
        # Нормализация
        lr_patch = self._normalize(lr_patch)
        hr_patch = self._normalize(hr_patch)
        
        # Конвертация в тензор и изменение порядка измерений (HWC -> CHW)
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).contiguous()
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).contiguous()
        
        return lr_tensor, hr_tensor


def create_datasets(
    train_lr_dir: str,
    train_hr_dir: str,
    valid_lr_dir: str,
    valid_hr_dir: str,
    patch_size: int = 32,
    scale: int = 2,
    normalize: str = '0-1'
) -> Tuple[Dataset, Dataset]:
    """
    Создает тренировочный и валидационный датасеты.
    
    Args:
        train_lr_dir (str): Папка с тренировочными LR изображениями
        train_hr_dir (str): Папка с тренировочными HR изображениями
        valid_lr_dir (str): Папка с валидационными LR изображениями
        valid_hr_dir (str): Папка с валидационными HR изображениями
        patch_size (int): Размер патча
        scale (int): Коэффициент масштабирования
        normalize (str): Тип нормализации
        
    Returns:
        tuple: (train_dataset, valid_dataset)
    """
    train_dataset = PatchDataset(
        lr_dir=train_lr_dir,
        hr_dir=train_hr_dir,
        patch_size=patch_size,
        scale=scale,
        augment=True,
        normalize=normalize
    )
    
    valid_dataset = PatchDataset(
        lr_dir=valid_lr_dir,
        hr_dir=valid_hr_dir,
        patch_size=patch_size,
        scale=scale,
        augment=False,  # Без аугментации для валидации
        normalize=normalize
    )
    
    return train_dataset, valid_dataset