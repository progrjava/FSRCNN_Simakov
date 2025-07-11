import os
import sys
from datetime import datetime

class Logger:
    """Класс для дублирования вывода в консоль и файл"""
    
    def __init__(self, log_file="results/training_output.txt"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Создаем файл с заголовком
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== ЛОГ ОБУЧЕНИЯ FSRCNN ===\n")
            f.write(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def print(self, message):
        """Печать в консоль и файл одновременно"""
        # В консоль
        print(message)
        
        # В файл
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')
    
    def close(self):
        """Закрытие лога"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nВремя завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")