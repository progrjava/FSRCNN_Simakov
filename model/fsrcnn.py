import torch
import torch.nn as nn
import torch.nn.init as init

class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Neural Network (FSRCNN)
    
    Args:
        scale_factor (int): Коэффициент увеличения разрешения (по умолчанию 2)
        num_channels (int): Количество входных каналов (по умолчанию 3 для RGB)
        d (int): Количество фильтров в слое извлечения признаков (по умолчанию 56)
        s (int): Количество фильтров в слоях сжатия (по умолчанию 12)
        m (int): Количество слоев отображения (по умолчанию 4)
    """
    
    def __init__(self, scale_factor=2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # 1. Извлечение признаков
        self.feature_extraction = nn.Conv2d(
            num_channels, d, kernel_size=5, padding=2
        )
        
        # 2. Сжатие
        self.shrinking = nn.Conv2d(d, s, kernel_size=1)
        
        # 3. Отображение (несколько сверточных слоев)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU())
        self.mapping = nn.Sequential(*mapping_layers)
        
        # 4. Расширение
        self.expanding = nn.Conv2d(s, d, kernel_size=1)
        
        # 5. Деконволюция (субпиксельный апсемплинг)
        self.deconv = nn.Conv2d(
            d, num_channels * (scale_factor ** 2), kernel_size=9, padding=4
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов сети"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.PReLU):
                init.constant_(module.weight, 0.25)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x (torch.Tensor): Входной LR тензор [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Выходной HR тензор [batch_size, channels, height*scale, width*scale]
        """
        # 1. Извлечение признаков
        x = torch.relu(self.feature_extraction(x))
        
        # 2. Сжатие
        x = torch.relu(self.shrinking(x))
        
        # 3. Отображение
        x = self.mapping(x)
        
        # 4. Расширение
        x = torch.relu(self.expanding(x))
        
        # 5. Деконволюция + субпиксельный апсемплинг
        x = self.deconv(x)
        x = self.pixel_shuffle(x)
        
        return x

def create_fsrcnn_model(scale_factor=2, num_channels=3):
    """
    Создает модель FSRCNN с предустановленными параметрами
    
    Args:
        scale_factor (int): Коэффициент увеличения
        num_channels (int): Количество каналов
        
    Returns:
        FSRCNN: Инициализированная модель
    """
    return FSRCNN(
        scale_factor=scale_factor,
        num_channels=num_channels,
        d=56,  # Количество фильтров извлечения признаков
        s=12,  # Количество фильтров сжатия
        m=4    # Количество слоев отображения
    )