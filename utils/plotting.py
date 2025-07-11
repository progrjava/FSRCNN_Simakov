import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def plot_training_curves(train_losses, valid_losses, train_psnrs, valid_psnrs, save_dir="results"):
    """Построение графиков обучения"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss графики
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss')
    ax1.grid(True)
    
    ax2.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.grid(True)
    
    # PSNR графики
    ax3.plot(epochs, train_psnrs, 'g-', label='Train PSNR')
    ax3.set_title('Training PSNR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)')
    ax3.grid(True)
    
    ax4.plot(epochs, valid_psnrs, 'orange', label='Validation PSNR')
    ax4.set_title('Validation PSNR')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('PSNR (dB)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(lr_img, hr_true, hr_pred, epoch, save_dir="results"):
    """Сравнение LR, HR истинного и предсказанного"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Конвертация тензоров в numpy
    lr_np = lr_img.cpu().permute(1, 2, 0).numpy()
    hr_true_np = hr_true.cpu().permute(1, 2, 0).numpy()
    hr_pred_np = hr_pred.cpu().permute(1, 2, 0).numpy()
    
    # Обрезка значений
    lr_np = np.clip(lr_np, 0, 1)
    hr_true_np = np.clip(hr_true_np, 0, 1)
    hr_pred_np = np.clip(hr_pred_np, 0, 1)
    
    axes[0].imshow(lr_np)
    axes[0].set_title(f'LR Input\n{lr_np.shape[1]}x{lr_np.shape[0]}')
    axes[0].axis('off')
    
    axes[1].imshow(hr_true_np)
    axes[1].set_title(f'HR Ground Truth\n{hr_true_np.shape[1]}x{hr_true_np.shape[0]}')
    axes[1].axis('off')
    
    axes[2].imshow(hr_pred_np)
    axes[2].set_title(f'HR Predicted\n{hr_pred_np.shape[1]}x{hr_pred_np.shape[0]}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_training_log(train_losses, valid_losses, train_psnrs, valid_psnrs, save_dir="results"):
    """Сохранение лога обучения в текстовый файл"""
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'training_log.txt'), 'w') as f:
        f.write("Epoch\tTrain_Loss\tValid_Loss\tTrain_PSNR\tValid_PSNR\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1}\t{train_losses[i]:.6f}\t{valid_losses[i]:.6f}\t{train_psnrs[i]:.2f}\t{valid_psnrs[i]:.2f}\n")