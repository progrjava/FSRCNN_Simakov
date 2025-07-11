import torch
import torch.nn.functional as F
import numpy as np

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Вычисляет PSNR между двумя изображениями
    
    Args:
        img1, img2 (torch.Tensor): Изображения для сравнения
        max_val (float): Максимальное значение пикселя
        
    Returns:
        float: PSNR в dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Упрощенная версия SSIM для быстрого вычисления
    
    Args:
        img1, img2 (torch.Tensor): Изображения для сравнения
        window_size (int): Размер окна
        max_val (float): Максимальное значение пикселя
        
    Returns:
        float: SSIM индекс
    """
    # Константы SSIM
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Средние значения
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Дисперсии и ковариация
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    # SSIM формула
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

class MetricsCalculator:
    """Класс для вычисления метрик качества изображений"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс накопленных метрик"""
        self.psnr_sum = 0
        self.ssim_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        """
        Обновление метрик новой парой изображений
        
        Args:
            pred (torch.Tensor): Предсказанное изображение
            target (torch.Tensor): Целевое изображение
        """
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            psnr = calculate_psnr(pred[i:i+1], target[i:i+1])
            ssim = calculate_ssim(pred[i:i+1], target[i:i+1])
            
            self.psnr_sum += psnr.item()
            self.ssim_sum += ssim.item()
            self.count += 1
    
    def get_metrics(self):
        """
        Получение средних метрик
        
        Returns:
            dict: Словарь с метриками
        """
        if self.count == 0:
            return {'psnr': 0, 'ssim': 0}
        
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count
        }