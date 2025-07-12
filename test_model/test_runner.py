import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Добавляем родительскую папку в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fsrcnn import create_fsrcnn_model
from utils.metrics import MetricsCalculator, calculate_psnr, calculate_ssim
from test_dataset import TestDataset

class ModelTester:
    def __init__(self, model_path="checkpoints/best_model.pth", device='cuda'):
        self.model_path = model_path
        self.device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
        self.model = None
        self.results_dir = "test_results"
        
        # Создание папки для результатов
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Устройство: {self.device}")
    
    def load_model(self):
        """Загрузка обученной модели"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        # Создание модели
        self.model = create_fsrcnn_model(scale_factor=2, num_channels=3)
        
        # Загрузка весов
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Эпоха обучения: {checkpoint.get('epoch', 'неизвестно')}")
        print(f"Валидационный PSNR: {checkpoint.get('valid_psnr', 'неизвестно'):.2f} dB")
        
        return checkpoint
    
    def create_test_dataset(self, test_lr_dir, test_hr_dir):
        """Создание тестового датасета"""
        test_dataset = TestDataset(
            lr_dir=test_lr_dir,
            hr_dir=test_hr_dir,
            scale=2,
            normalize='0-1'
        )
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return test_loader
    
    def save_image_results(self, lr_img, hr_true, hr_pred, filename, psnr, ssim, idx):
        """Сохранение всех результатов для одного изображения"""
        # Создание папки для изображения
        img_name = filename.split('.')[0]
        img_dir = os.path.join(self.results_dir, f"image_{idx+1:02d}_{img_name}")
        os.makedirs(img_dir, exist_ok=True)
        
        # Конвертация тензоров в numpy
        lr_np = lr_img.cpu().permute(1, 2, 0).numpy()
        hr_true_np = hr_true.cpu().permute(1, 2, 0).numpy()
        hr_pred_np = hr_pred.cpu().permute(1, 2, 0).numpy()
        
        # Обрезка значений до [0, 1]
        lr_np = np.clip(lr_np, 0, 1)
        hr_true_np = np.clip(hr_true_np, 0, 1)
        hr_pred_np = np.clip(hr_pred_np, 0, 1)
        
        # Сохранение отдельных изображений с оригинальными размерами через PIL
        lr_img_pil = Image.fromarray((lr_np * 255).astype(np.uint8))
        lr_img_pil.save(os.path.join(img_dir, 'LR_input.png'))
        
        hr_true_img_pil = Image.fromarray((hr_true_np * 255).astype(np.uint8))
        hr_true_img_pil.save(os.path.join(img_dir, 'HR_ground_truth.png'))
        
        hr_pred_img_pil = Image.fromarray((hr_pred_np * 255).astype(np.uint8))
        hr_pred_img_pil.save(os.path.join(img_dir, 'HR_predicted.png'))
        
        # Сохранение сравнительного изображения
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(lr_np)
        axes[0].set_title(f'LR Input\n{lr_np.shape[1]}x{lr_np.shape[0]}')
        axes[0].axis('off')
        
        axes[1].imshow(hr_true_np)
        axes[1].set_title(f'HR Ground Truth\n{hr_true_np.shape[1]}x{hr_true_np.shape[0]}')
        axes[1].axis('off')
        
        axes[2].imshow(hr_pred_np)
        axes[2].set_title(f'HR Predicted\n{hr_pred_np.shape[1]}x{hr_pred_np.shape[0]}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_test(self, test_lr_dir="data/test/low_resolution", 
                 test_hr_dir="data/test/high_resolution", 
                 save_images=True, max_save_images=5):
        """Запуск тестирования модели"""
        
        # Загрузка модели
        checkpoint = self.load_model()
        
        # Создание датасета
        print("\nЗагрузка тестовых данных...")
        test_loader = self.create_test_dataset(test_lr_dir, test_hr_dir)
        
        # Инициализация калькулятора метрик
        metrics_calc = MetricsCalculator()
        
        # Списки для детальных результатов
        detailed_results = []
        
        print(f"\nНачало тестирования на {len(test_loader)} изображениях...")
        start_time = time.time()
        
        with torch.no_grad():
            for idx, (lr_img, hr_img, filename) in enumerate(test_loader):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)
                
                # Генерация SR изображения
                sr_img = self.model(lr_img)
                
                # Вычисление метрик
                psnr_value = calculate_psnr(sr_img, hr_img).item()
                ssim_value = calculate_ssim(sr_img, hr_img).item()
                
                # Обновление общих метрик
                metrics_calc.update(sr_img, hr_img)
                
                # Сохранение детальных результатов
                detailed_results.append({
                    'filename': filename[0],
                    'psnr': psnr_value,
                    'ssim': ssim_value
                })
                
                print(f"Изображение {idx+1}/{len(test_loader)}: {filename[0]} - "
                      f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
                
                # Сохранение результатов для изображения
                if save_images and idx < max_save_images:
                    self.save_image_results(
                        lr_img[0], hr_img[0], sr_img[0], 
                        filename[0], psnr_value, ssim_value, idx
                    )
        
        # Получение средних метрик
        avg_metrics = metrics_calc.get_metrics()
        test_time = time.time() - start_time
        
        # Вывод результатов
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print(f"Средний PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"Средний SSIM: {avg_metrics['ssim']:.4f}")
        print(f"Время тестирования: {test_time:.1f}s")
        print(f"Время на изображение: {test_time/len(test_loader):.2f}s")
        print("="*60)
        
        # Сохранение детальных результатов
        self.save_detailed_results(detailed_results, avg_metrics, test_time, checkpoint)
        
        return avg_metrics, detailed_results
    
    def save_detailed_results(self, detailed_results, avg_metrics, test_time, checkpoint):
        """Сохранение детальных результатов в файл"""
        results_file = os.path.join(self.results_dir, "test_metrics.txt")
        
        with open(results_file, "w", encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ FSRCNN\n")
            f.write("="*50 + "\n\n")
            
            # Информация о модели
            f.write("ИНФОРМАЦИЯ О МОДЕЛИ:\n")
            f.write(f"Путь к модели: {self.model_path}\n")
            f.write(f"Эпоха обучения: {checkpoint.get('epoch', 'неизвестно')}\n")
            f.write(f"Валидационный PSNR: {checkpoint.get('valid_psnr', 'неизвестно'):.2f} dB\n")
            f.write(f"Устройство: {self.device}\n\n")
            
            # Общие результаты
            f.write("ОБЩИЕ РЕЗУЛЬТАТЫ:\n")
            f.write(f"Количество тестовых изображений: {len(detailed_results)}\n")
            f.write(f"Средний PSNR: {avg_metrics['psnr']:.2f} dB\n")
            f.write(f"Средний SSIM: {avg_metrics['ssim']:.4f}\n")
            f.write(f"Время тестирования: {test_time:.1f}s\n")
            f.write(f"Время на изображение: {test_time/len(detailed_results):.2f}s\n\n")
            
            # Детальные результаты по изображениям
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
            f.write("-"*50 + "\n")
            for i, result in enumerate(detailed_results, 1):
                f.write(f"{i:2d}. {result['filename']:20s} | "
                       f"PSNR: {result['psnr']:6.2f} dB | "
                       f"SSIM: {result['ssim']:.4f}\n")
        
        print(f"\nДетальные результаты сохранены в: {results_file}")