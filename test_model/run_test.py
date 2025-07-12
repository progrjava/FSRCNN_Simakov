"""
Простой скрипт для запуска тестирования модели FSRCNN
"""

from test_runner import ModelTester

def main():
    print("Запуск тестирования модели FSRCNN")
    print("=" * 50)
    
    # Создание тестера
    tester = ModelTester(
        model_path="checkpoints/best_model.pth",
        device='cuda'
    )
    
    try:
        # Запуск тестирования
        avg_metrics, detailed_results = tester.run_test(
            test_lr_dir="data/test/low_resolution",
            test_hr_dir="data/test/high_resolution",
            save_images=True,
            max_save_images=15
        )
        
        print("\nТестирование завершено успешно!")
        print(f"Средний PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"Средний SSIM: {avg_metrics['ssim']:.4f}")
        print(f"Результаты сохранены в папке: test_results/")
        
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()