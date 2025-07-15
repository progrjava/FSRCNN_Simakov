import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_metrics(log_file="training_results/training_log.txt", save_dir="training_results"):
    """
    Построение графиков метрик обучения из лог файла
    """
    # Чтение данных из файла
    data = pd.read_csv(log_file, sep='\t')
    
    epochs = data['Epoch']
    train_loss = data['Train_Loss']
    valid_loss = data['Valid_Loss']
    train_psnr = data['Train_PSNR']
    valid_psnr = data['Valid_PSNR']
    
    # Создание фигуры с 4 подграфиками
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # График Train Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L1 Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, max(epochs))
    
    # График Valid Loss с трендом
    ax2.plot(epochs, valid_loss, 'r-', linewidth=2, marker='s', markersize=4)
    z = np.polyfit(epochs, valid_loss, 1)
    p = np.poly1d(z)
    ax2.plot(epochs, p(epochs), 'k--', alpha=0.8, linewidth=1.5, label=f'Тренд (k={z[0]:.6f})')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(1, max(epochs))
    
    # График Train PSNR
    ax3.plot(epochs, train_psnr, 'g-', linewidth=2, marker='^', markersize=4)
    ax3.set_title('Training PSNR', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, max(epochs))
    
    # График Valid PSNR с трендом
    ax4.plot(epochs, valid_psnr, 'orange', linewidth=2, marker='d', markersize=4)
    z = np.polyfit(epochs, valid_psnr, 1)
    p = np.poly1d(z)
    ax4.plot(epochs, p(epochs), 'k--', alpha=0.8, linewidth=1.5, label=f'Тренд (k={z[0]:.4f})')
    ax4.set_title('Validation PSNR', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('PSNR (dB)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(1, max(epochs))
    
    # Настройка общего вида
    plt.tight_layout(pad=3.0)
    
    # Сохранение графика
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_training_metrics()