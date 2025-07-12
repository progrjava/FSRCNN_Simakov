import torch
import os
import cv2
import numpy as np
from PIL import Image
import sys

# Добавляем путь для импорта модели
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.fsrcnn import create_fsrcnn_model

class ImageEnhancer:
    def __init__(self, model_path="checkpoints/best_model.pth", device='cuda'):
        self.device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_path = model_path
        
        print(f"Устройство: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        # Создание и загрузка модели
        self.model = create_fsrcnn_model(scale_factor=2, num_channels=3)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Модель загружена из {self.model_path}")
    
    def preprocess_image(self, image_path):
        """Предобработка изображения для модели"""
        # Загрузка изображения
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Не удалось загрузить изображение: {image_path}")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Нормализация в диапазон [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Конвертация в тензор (HWC -> CHW) и добавление batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    def postprocess_image(self, tensor):
        """Постобработка результата модели"""
        # Удаление batch dimension и конвертация CHW -> HWC
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Обрезка значений до [0, 1] и конвертация в uint8
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def enhance_image(self, image_path):
        """Увеличение разрешения одного изображения"""
        # Предобработка
        lr_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Генерация HR изображения
        with torch.no_grad():
            hr_tensor = self.model(lr_tensor)
        
        # Постобработка
        hr_image = self.postprocess_image(hr_tensor)
        
        return hr_image
    
    def enhance_folder(self, input_dir="real_images/LR_images", output_dir="real_images/HR_images"):
        """Увеличение разрешения всех изображений в папке"""
        # Проверка существования папок
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Входная папка не найдена: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Получение списка изображений
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"В папке {input_dir} не найдено изображений")
            return
        
        print(f"Найдено {len(image_files)} изображений для обработки")
        
        # Обработка каждого изображения
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            
            # Создание имени выходного файла
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_enhanced{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                print(f"Обработка {i}/{len(image_files)}: {filename}")
                
                # Увеличение разрешения
                hr_image = self.enhance_image(input_path)
                
                # Сохранение результата
                hr_pil = Image.fromarray(hr_image)
                hr_pil.save(output_path)
                
                print(f"  Сохранено: {output_filename}")
                
            except Exception as e:
                print(f"  Ошибка при обработке {filename}: {e}")
        
        print(f"\nОбработка завершена! Результаты сохранены в {output_dir}")

def main():
    """Основная функция"""
    try:
        enhancer = ImageEnhancer(
            model_path="checkpoints/best_model.pth",
            device='cuda'
        )
        
        enhancer.enhance_folder(
            input_dir="real_images/LR_images",
            output_dir="real_images/HR_images"
        )
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()