import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.fsrcnn import create_fsrcnn_model
from utils.data_loader import create_datasets
from utils.plotting import plot_training_curves, plot_comparison, save_training_log
from utils.logger import Logger
import os
import time

def calculate_psnr(img1, img2):
    """Вычисляет PSNR между двумя изображениями"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train_model(num_epochs=50, batch_size=32, learning_rate=0.001, device='cuda'):
    """
    Обучение модели FSRCNN
    
    Args:
        num_epochs (int): Количество эпох
        batch_size (int): Размер батча
        learning_rate (float): Скорость обучения
        device (str): Устройство для обучения
    """
    # Проверка устройства
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Инициализация логгера
    logger = Logger("training_results/training_output.txt")
    
    if device == 'cpu':
        logger.print("CUDA недоступна, используется CPU")
    logger.print(f"Устройство: {device}")
    
    # Создание датасетов
    logger.print("Загрузка данных...")
    train_dataset, valid_dataset = create_datasets(
        train_lr_dir="data/train/low_resolution",
        train_hr_dir="data/train/high_resolution", 
        valid_lr_dir="data/valid/low_resolution",
        valid_hr_dir="data/valid/high_resolution",
        patch_size=48,
        scale=2,
        normalize='0-1'
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.print(f"Тренировочных батчей: {len(train_loader)}")
    logger.print(f"Валидационных батчей: {len(valid_loader)}")
    
    # Модель
    model = create_fsrcnn_model(scale_factor=2, num_channels=3)
    model = model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Папки для сохранения
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("training_results", exist_ok=True)
    
    best_psnr = 0
    
    # Списки для графиков
    train_losses = []
    valid_losses = []
    train_psnrs = []
    valid_psnrs = []
    
    logger.print(f"\nНачало обучения на {num_epochs} эпох...")
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0
        train_psnr = 0
        start_time = time.time()
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            sr_batch = model(lr_batch)
            
            # Вычисление потерь
            loss = criterion(sr_batch, hr_batch)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_psnr += calculate_psnr(sr_batch, hr_batch).item()
            
            if batch_idx % 50 == 0:
                logger.print(f'Эпоха {epoch+1}/{num_epochs}, Батч {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # Валидация
        model.eval()
        valid_loss = 0
        valid_psnr = 0
        
        with torch.no_grad():
            for lr_batch, hr_batch in valid_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                sr_batch = model(lr_batch)
                loss = criterion(sr_batch, hr_batch)
                
                valid_loss += loss.item()
                valid_psnr += calculate_psnr(sr_batch, hr_batch).item()
        
        # Средние значения
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        valid_psnr /= len(valid_loader)
        train_psnr /= len(train_loader)
        
        # Сохранение для графиков
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_psnrs.append(train_psnr)
        valid_psnrs.append(valid_psnr)
        
        epoch_time = time.time() - start_time
        
        logger.print(f'Эпоха {epoch+1}/{num_epochs}:')
        logger.print(f'  Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.2f} dB')
        logger.print(f'  Valid Loss: {valid_loss:.6f}, Valid PSNR: {valid_psnr:.2f} dB')
        logger.print(f'  Время: {epoch_time:.1f}s')
        
        # Сохранение лучшей модели
        if valid_psnr > best_psnr:
            best_psnr = valid_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_psnr': valid_psnr,
            }, 'checkpoints/best_model.pth')
            logger.print(f'Новая лучшая модель сохранена (PSNR: {best_psnr:.2f})')
        
        # Сохранение примера каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(valid_loader))
                sample_lr = sample_lr[:1].to(device)  # Первый образец
                sample_hr = sample_hr[:1].to(device)
                sample_sr = model(sample_lr)
                
                plot_comparison(
                    sample_lr[0], sample_hr[0], sample_sr[0], 
                    epoch + 1, "training_results"
                )
        
        logger.print('-' * 50)
    
    # Сохранение финальных результатов
    plot_training_curves(train_losses, valid_losses, train_psnrs, valid_psnrs, "training_results")
    save_training_log(train_losses, valid_losses, train_psnrs, valid_psnrs, "training_results")
    
    logger.print(f"Обучение завершено! Лучший PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    train_model(
        num_epochs=30,
        batch_size=16,
        learning_rate=0.001,
        device='cuda'
    )